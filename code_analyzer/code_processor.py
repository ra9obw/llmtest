import os
from pathlib import Path
import traceback 
from typing import List, Dict, Any, Optional, Callable, Set
from functools import partial
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TypeKind, StorageClass
from interfaces import IElementTracker, IFileProcessor, IJsonDataStorage
from element_tracker import ElementTracker
from file_processor import FileProcessor
from structure_models.storage import JsonDataStorage
from code_processing.range_locator import RangeLocator
from code_processing.template_extractor import TemplateBodyExtractor

class CodeExtractor:
    """Extracts code structures from C++ source files using flat storage structure."""
    
    def __init__(self, 
                 repo_path: str, 
                 skip_files_func: Optional[Callable[[str], bool]] = None,
                 element_tracker: Optional[IElementTracker] = None,
                 file_processor: Optional[IFileProcessor] = None,
                 data_storage: Optional[IJsonDataStorage] = None,
                 log_level: int = 0) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self.skip_files_func = skip_files_func
        
        self.file_processor = file_processor or FileProcessor(self.repo_path)
        self.element_tracker = element_tracker or ElementTracker(partial(self.file_processor.get_relative_path))
        self.data_storage = data_storage or JsonDataStorage()

        self.log_level = log_level
        self.log_level_desc = {
            0: "note",
            1: "warn",
            2: "err",
            3: "crit"
        }
        self._cursor_handlers = self._init_cursor_handlers()
        self._RELEVANT_CURSOR_KINDS = set(self._cursor_handlers.keys())
        self.data_storage.create_element_storage(set([cr.name.lower() for cr in self._cursor_handlers.keys()]))

        self.range_locator = RangeLocator(self._RELEVANT_CURSOR_KINDS, self.file_processor)
        self.template_extractor = TemplateBodyExtractor(self.range_locator)

    def log(self, msg, level=0):
        if level >= self.log_level:
            print(f"{self.log_level_desc.get(level, "")}\t{msg}")

    def _init_cursor_handlers(self) -> Dict[CursorKind, Callable[[Any], None]]:
        return {
            CursorKind.CLASS_DECL: self._process_class,
            CursorKind.STRUCT_DECL: self._process_class,
            CursorKind.CLASS_TEMPLATE: self._process_class_template,
            CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: self._process_class_template,
            CursorKind.NAMESPACE: self._process_namespace,
            CursorKind.CXX_METHOD: self._process_method,
            CursorKind.CONSTRUCTOR: self._process_method,
            CursorKind.DESTRUCTOR: self._process_method,
            CursorKind.FUNCTION_DECL: self._process_function,
            CursorKind.FUNCTION_TEMPLATE: self._process_function_template,
            CursorKind.LAMBDA_EXPR: self._process_lambda,
            CursorKind.PREPROCESSING_DIRECTIVE: self._process_preprocessor_directive,
            CursorKind.MACRO_DEFINITION: self._process_macro,
            CursorKind.CONVERSION_FUNCTION: self._process_literal,
            CursorKind.TYPE_ALIAS_DECL: self._process_type_alias,
            CursorKind.TYPEDEF_DECL: self._process_typedef,
            CursorKind.VAR_DECL: self._process_variable,
            CursorKind.FIELD_DECL: self._process_field,
            CursorKind.ENUM_DECL: self._process_enum,
            CursorKind.ENUM_CONSTANT_DECL: self._process_enum_constant,
            CursorKind.UNION_DECL: self._process_union,
            CursorKind.USING_DIRECTIVE: self._process_using_directive
        }

    def _get_comments_before_cursor(self, cursor) -> Dict[str, Any]:
        """Extracts comments and docstrings before the cursor position."""
        if not cursor.location.file:
            return {"comments": [], "docstrings": []}
        
        file_path = cursor.location.file.name
        start_line = cursor.location.line
        
        # Get the previous significant cursor position to determine search range
        prev_cursor_pos = self.range_locator.get_previous_cursor_position(cursor)
        
        if not prev_cursor_pos:
            search_start = 1  # Start from beginning of file if no previous cursor
        else:
            search_start = prev_cursor_pos["line"]
        
        # Extract comments between previous cursor and current cursor
        comments = []
        docstrings = []
        
        # Get all comments in the file
        all_comments = self.range_locator.get_comments_in_range(
            file_path, 
            start_line=search_start,
            end_line=start_line - 1  # Look before current cursor
        )
        
        # Classify comments as regular comments or docstrings
        for comment in all_comments:
            comment_text = comment["text"].strip()
            if comment_text.startswith(("/**", "/*!", "///", "//!")):
                docstrings.append(comment)
            else:
                comments.append(comment)
        
        return {
            "comments": comments,
            "docstrings": docstrings
        }

    def _get_function_signature(self, cursor) -> Dict[str, Any]:
        signature = {
            "name": cursor.spelling,
            "return_type": cursor.result_type.spelling,
            "parameters": [],
            "qualifiers": {
                "is_const": cursor.is_const_method(),
                "is_static": cursor.is_static_method(),
                "is_virtual": cursor.is_virtual_method(),
                "is_pure_virtual": cursor.is_pure_virtual_method(),
                "is_noexcept": "noexcept" in cursor.result_type.spelling,
            }
        }

        for i, arg in enumerate(cursor.get_arguments()):
            param = {
                "name": arg.spelling or f"param_{i}",
                "type": arg.type.spelling,
                "default_value": self._get_default_value(arg)
            }
            signature["parameters"].append(param)
        
        return signature

    def _get_default_value(self, arg) -> Optional[str]:
        for child in arg.get_children():
            if child.kind == CursorKind.UNEXPOSED_EXPR:
                return self.range_locator.get_code_snippet(child)
        return None

    def _get_parent_info(self, cursor):
        parent = cursor.semantic_parent
        if not parent:
            return None, None, None
        
        parent_id = self.element_tracker.generate_element_id(parent)
        parent_type = parent.kind.name.lower()
        parent_name = parent.spelling or "(anonymous)"
        
        return parent_id, parent_type, parent_name

    def _process_type_alias(self, cursor) -> None:
        """Обрабатывает объявления типа через using (type aliases)"""
        if self._is_inside_class_or_function(cursor):
            return True
            
        self.log(f"{cursor.kind}:\t_process_type_alias {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
        
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)
        
        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            # "underlying_type": cursor.underlying_typedef_type.spelling if cursor.underlying_typedef_type else "unknown",
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "code": self.range_locator.get_code_snippet(cursor),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True
    
    def _process_union(self, cursor) -> None:
        """Обрабатывает объявления union"""
        if self._is_inside_class_or_function(cursor):
            return True
        if not cursor.is_definition():
            return True

        self.log(f"{cursor.kind}:\t_process_union {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "code": self.range_locator.get_code_snippet(cursor),
            "size": cursor.type.get_size(),
            "fields": self._get_union_fields(cursor),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True
    
    def _get_union_fields(self, cursor) -> List[Dict]:
        """Извлекает информацию о полях union"""
        fields = []
        for child in cursor.get_children():
            if child.kind == CursorKind.FIELD_DECL:
                fields.append({
                    "name": child.spelling,
                    "type": child.type.spelling,
                    "size": child.type.get_size(),
                    "line": child.location.line
                })
        return fields

    def _process_using_directive(self, cursor) -> None:
        """Обрабатывает директивы using для пространств имен"""
        self.log(f"{cursor.kind}:\t_process_using_directive in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        # Получаем имя импортируемого пространства имен
        namespace = None
        for child in cursor.get_children():
            if child.kind == CursorKind.NAMESPACE_REF:
                namespace = child.spelling
                break

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "namespace": namespace,
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "code": self.range_locator.get_code_snippet(cursor),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True
           
    def _process_typedef(self, cursor) -> None:
        """Обрабатывает typedef объявления"""
        if self._is_inside_class_or_function(cursor):
            return True
            
        self.log(f"{cursor.kind}:\t_process_typedef {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
        
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)
        
        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            # "underlying_type": cursor.underlying_typedef_type.spelling if cursor.underlying_typedef_type else "unknown",
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "code": self.range_locator.get_code_snippet(cursor),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_variable(self, cursor) -> None:
        """Обрабатывает объявления переменных (глобальных и в пространствах имен)"""
        if self._is_inside_class_or_function(cursor):
            return True
            
        self.log(f"{cursor.kind}:\t_process_variable {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
        
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)
        
        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "var_type": cursor.type.spelling,
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "code": self.range_locator.get_code_snippet(cursor),
            "is_const": cursor.type.is_const_qualified(),
            "is_static": cursor.storage_class == StorageClass.STATIC,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_field(self, cursor) -> None:
        """Обрабатывает поля классов/структур (уже обрабатываются в _process_class)"""
        return True

    def _process_enum(self, cursor) -> None:
        """Обрабатывает объявления enum"""
        if self._is_inside_class_or_function(cursor):
            return True
            
        self.log(f"{cursor.kind}:\t_process_enum {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
        
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)
        
        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "code": self.range_locator.get_code_snippet(cursor),
            "is_scoped": cursor.is_scoped_enum(),
            "underlying_type": cursor.enum_type.spelling if cursor.enum_type else "int",
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_enum_constant(self, cursor) -> None:
        """Обрабатывает константы enum (уже обрабатываются в _process_enum)"""
        return False

    def _is_inside_class_or_function(self, cursor) -> bool:
        """Проверяет, находится ли курсор внутри класса/структуры или функции"""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind in (CursorKind.CLASS_DECL,       CursorKind.STRUCT_DECL, 
                               CursorKind.CLASS_TEMPLATE,   CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
                               CursorKind.FUNCTION_DECL,    CursorKind.FUNCTION_TEMPLATE, 
                               CursorKind.CXX_METHOD,       CursorKind.LAMBDA_EXPR):
                return True
            parent = parent.semantic_parent
        return False

    def _process_class(self, cursor) -> None:
        if not cursor.is_definition():
            self.log(f"!!not cursor.is_definition():\t_process_class {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
            return

        self.log(f"{cursor.kind.name}:\t_process_class {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "column": cursor.location.column,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return False

    def _process_class_template(self, cursor) -> None:
        self.log(f"{cursor.kind.name}:\t_process_class_template {cursor.spelling or f"anon_template_{cursor.location.line}"} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "full_body": self.template_extractor.get_template_method_body(cursor),
            "template_parameters": self._get_template_parameters(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return False

    def _process_method(self, cursor) -> None:
        
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        
        self.log(f"{cursor.kind.name}:\t_process_method {cursor.spelling} of {parent_name} in {self.file_processor.get_relative_path(cursor.location.file.name)}", 3)

        code = self.range_locator.get_code_snippet(cursor)
        is_defined = cursor.is_definition()
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "signature": self._get_function_signature(cursor),
            "code": code,
            "is_defined": str(is_defined),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "is_template": parent_type in ("class_template", "class_template_partial_specialization"),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_function(self, cursor) -> None:

        self.log(f"{cursor.kind}:\t_process_method {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        is_defined = cursor.is_definition()
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "signature": self._get_function_signature(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "is_defined": str(is_defined),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_function_template(self, cursor) -> None:    
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        self.log(f"{cursor.kind.name}:\t_process_function_template {cursor.spelling} of {parent_type} : {parent_name} in {self.file_processor.get_relative_path(cursor.location.file.name)}", 3)
        
        code = self.range_locator.get_code_snippet(cursor)
        is_defined = cursor.is_definition()
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)
        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "signature": self._get_function_signature(cursor),
            "code": code,
            "is_defined": str(is_defined),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "is_template": True,
            "template_parameters": self._get_template_parameters(cursor),
            "context_before": context["context_before"],
            "context_after": context["context_after"],
        "comments": comments["comments"],
        "docstrings": comments["docstrings"]
        }
        self.data_storage.add_element(element["type"], element)

        return True

    def _process_namespace(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        _name = self.element_tracker.generate_name(cursor)
        self.log(f"{cursor.kind.name}:\t_process_namespace {_name} of {parent_type} : {parent_name} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": _name,
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        self.data_storage.add_element(element["type"], element)
        return False

    def _process_lambda(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        _name = self.element_tracker.generate_name(cursor)
        self.log(f"{cursor.kind.name}:\t_process_lambda {_name} of {parent_type} : {parent_name} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": _name,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_preprocessor_directive(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_macro(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        self.data_storage.add_element(element["type"], element)
        return True

    def _process_literal(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        context = self.range_locator.get_context(cursor)
        comments = self._get_comments_before_cursor(cursor)

        element = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": cursor.kind.name.lower(),
            "name": self.element_tracker.generate_name(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "context_before": context["context_before"],
            "context_after": context["context_after"],
            "comments": comments["comments"],
            "docstrings": comments["docstrings"]
        }
        self.data_storage.add_element(element["type"], element)
        return True

    def _get_template_parameters(self, cursor) -> List[str]:
        params = []
        for child in cursor.get_children():
            if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                params.append(child.spelling)
        return params

    def visit_node(self, cursor):
        if cursor.kind == CursorKind.TRANSLATION_UNIT:
            for child in cursor.get_children():
                self.visit_node(child)
            return

        if self._should_skip_cursor(cursor):
            return

        cursor_id = self.element_tracker.generate_element_id(cursor)
        if not self.element_tracker.is_processed(cursor_id):
            if cursor.kind in self._RELEVANT_CURSOR_KINDS:
                self.element_tracker.mark_processed(cursor_id)
                handler = self._cursor_handlers.get(cursor.kind)
                if handler:
                    if handler(cursor): #if there is no interesting nested elements, then return
                        return
            else:
                self.element_tracker.track_unhandled_kind(cursor)
                # print(cursor.kind.name, cursor.spelling)

            for child in cursor.get_children():
                self.visit_node(child)
        else:
            self.log(f"\tElement is processed:\t{cursor.kind.name}, {cursor.spelling}, {self.file_processor.get_relative_path(cursor.location.file.name)}")

    def _should_skip_cursor(self, cursor) -> bool:
        file_path = cursor.location.file.name if cursor.location.file else None
        if not file_path:
            return True
        return (self.file_processor.is_system_header(file_path) or \
               (self.skip_files_func and self.skip_files_func(self.file_processor.get_relative_path(file_path))))

    def process_file(self, file_path: Path) -> None:
        translation_unit = self.file_processor.parse_file(file_path, self.skip_files_func)
        if translation_unit:
            self.visit_node(translation_unit.cursor)

    @property
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        return self.element_tracker.unprocessed_stats

def main() -> None:
    from settings import settings
    Config.set_library_file(settings["CLANG_PATH"])
    BASE_ROOT = settings["BASE_ROOT"]
    # PROJ_NAME = settings["PROJ_NAME"]
    # PROJ_NAME = r"simple"
    # PROJ_NAME = r"adc4x250"
    # PROJ_NAME = r"cppTango-9.3.7"
    PROJ_NAME = r"template_exampl"
    # PROJ_NAME = r"template_test_simple"
    # PROJ_NAME = r"ifdef_example"
    REPO_PATH = os.path.join(BASE_ROOT, "codebase", PROJ_NAME)
    OUTPUT_JSONL = os.path.join(BASE_ROOT, f"dataset_clang_{PROJ_NAME}.jsonl")

    # tango_generated_stop_list = [
    #     "main.cpp",
    #     "*Class.cpp",
    #     "*Class.h",
    #     "*DynAttrUtils.cpp",
    #     "*StateMachine.cpp",
    #     "ClassFactory.cpp",
    # ]

    # def should_skip_file(filename):
    #     """Check if file should be skipped based on stop list"""
    #     for pattern in tango_generated_stop_list:
    #         if pattern.startswith('*'):
    #             # Pattern matches end of filename
    #             if filename.endswith(pattern[1:]):
    #                 return True
    #         else:
    #             # Exact match
    #             if filename == pattern:
    #                 return True
    #     return False


    tango_codebase_stop_list = [
        "cpp_test_suite"
    ]
    def should_skip_file(filename):
        for pattern in tango_codebase_stop_list:
            if pattern in filename:
                    return True
        return False


    data_storage = JsonDataStorage(OUTPUT_JSONL)
    extractor = CodeExtractor(REPO_PATH, data_storage=data_storage, skip_files_func=should_skip_file, log_level=3)

    for root, _, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx", ".tpp")):
                file_path = Path(root) / file
                print(f"Processing: {file_path}")
                try:
                    extractor.process_file(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    traceback.print_exc()
    data_storage.print_statistics(unprocessed_stats = extractor.unprocessed_stats )
    data_storage.save_to_file()
    print(f"Results saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()