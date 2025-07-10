import os
from pathlib import Path
import traceback 
from typing import List, Dict, Any, Optional, Callable, Set
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TypeKind
from interfaces import IElementTracker, IFileProcessor
from element_tracker import ElementTracker
from file_processor import FileProcessor
from structure_models.storage import JsonDataStorage
from code_processing.range_locator import RangeLocator
from code_processing.template_extractor import TemplateBodyExtractor

class CodeExtractor:
    """Extracts code structures (classes, functions, templates, etc.) from C++ source files."""
    
    def __init__(self, 
                 repo_path: str, 
                 skip_files_func: Optional[Callable[[str], bool]] = None,
                 element_tracker: Optional[IElementTracker] = None,
                 file_processor: Optional[IFileProcessor] = None,
                 data_storage: Optional[JsonDataStorage] = None,
                 log_level: int = 0) -> None:
        """Initialize the code extractor with repository path and optional dependencies."""
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self.skip_files_func = skip_files_func
        
        # Initialize dependencies
        self.element_tracker = element_tracker or ElementTracker()
        self.file_processor = file_processor or FileProcessor(self.repo_path)
        self.data_storage = data_storage or JsonDataStorage()

        self.log_level = log_level
        # Cursor kinds we're interested in for position tracking
        self._cursor_handlers = self._init_cursor_handlers()
        self._RELEVANT_CURSOR_KINDS = set(self._cursor_handlers.keys())

        # Initialize helper classes
        self.range_locator = RangeLocator(self._RELEVANT_CURSOR_KINDS, self.file_processor)
        self.template_extractor = TemplateBodyExtractor(self.range_locator)

    def _init_cursor_handlers(self) -> Dict[CursorKind, Callable[[Any], None]]:
        """Initialize the dictionary of cursor kind to handler function mappings."""
        return {
            # Classes/structs
            CursorKind.CLASS_DECL: self._process_class,
            CursorKind.STRUCT_DECL: self._process_class,
            # Class templates
            CursorKind.CLASS_TEMPLATE: self._process_class_template,
            CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: self._process_class_template,
            # Namespaces
            CursorKind.NAMESPACE: self._process_namespace,
            # Methods and functions
            CursorKind.CXX_METHOD: self._process_class_member,
            CursorKind.FUNCTION_DECL: self._process_function,
            # Function templates
            CursorKind.FUNCTION_TEMPLATE: self._process_template,
            # Lambdas
            CursorKind.LAMBDA_EXPR: self._process_lambda,
            # Preprocessor
            CursorKind.PREPROCESSING_DIRECTIVE: self._process_preprocessor_directive,
            # Error handling
            CursorKind.CXX_TRY_STMT: self._process_error_handler,
            # Macros
            CursorKind.MACRO_DEFINITION: self._process_macro,
            # Literals
            CursorKind.CONVERSION_FUNCTION: self._process_literal,
            # Attributes
            CursorKind.ANNOTATE_ATTR: self._process_attribute,
        }

    def log(self, _str, severity = 'note'):
        if severity == 'critical':
            print(_str)
        elif severity == 'note' and self.log_level >= 1:
            print(_str)

    def _get_relative_path(self, absolute_path: str) -> str:
        """Convert absolute path to relative path from repo root."""
        try:
            return str(Path(absolute_path).resolve().relative_to(self.repo_path))
        except ValueError:
            return absolute_path

    def _get_function_signature(self, cursor) -> Dict[str, Any]:
        """Extract detailed function signature information."""
        signature = {
            "name": cursor.spelling,
            "return_type": cursor.result_type.spelling,
            "parameters": [],
            "is_const": False,
            "is_static": False,
            "is_virtual": False,
            "is_pure_virtual": False,
            "is_override": False,
            "is_final": False,
            "is_noexcept": False,
            "is_explicit": False,
            "is_volatile": False,
            "storage_class": None,
            "calling_convention": None
        }

        # Process function qualifiers
        if cursor.is_const_method():
            signature["is_const"] = True
        if cursor.is_static_method():
            signature["is_static"] = True
        if cursor.is_virtual_method():
            signature["is_virtual"] = True
        if cursor.is_pure_virtual_method():
            signature["is_pure_virtual"] = True
        
        # Alternative way to check override and final
        try:
            if hasattr(cursor, 'is_override_method') and cursor.is_override_method():
                signature["is_override"] = True
            if hasattr(cursor, 'is_final_method') and cursor.is_final_method():
                signature["is_final"] = True
        except:
            type_spelling = cursor.type.spelling
            signature["is_override"] = "override" in type_spelling
            signature["is_final"] = "final" in type_spelling
        
        # Process parameters
        for i, arg in enumerate(cursor.get_arguments()):
            param = {
                "name": arg.spelling or f"param_{i}",
                "type": arg.type.spelling,
                "type_kind": str(arg.type.kind),
                "is_const": arg.type.is_const_qualified(),
                "is_volatile": arg.type.is_volatile_qualified(),
                "is_reference": arg.type.kind == TypeKind.LVALUEREFERENCE,
                "is_pointer": arg.type.kind == TypeKind.POINTER,
                "default_value": None
            }
            
            # Try to get default value if exists
            for child in arg.get_children():
                if child.kind == CursorKind.UNEXPOSED_EXPR:
                    param["default_value"] = self.range_locator.get_code_snippet(child)
                    break
            
            signature["parameters"].append(param)
        
        # Detect noexcept
        signature["is_noexcept"] = (
            "noexcept" in cursor.result_type.spelling or 
            any(tok.spelling == "noexcept" for tok in cursor.get_tokens())
        )
        
        # Detect calling convention
        if "__stdcall" in cursor.result_type.spelling:
            signature["calling_convention"] = "stdcall"
        elif "__fastcall" in cursor.result_type.spelling:
            signature["calling_convention"] = "fastcall"
        elif "__cdecl" in cursor.result_type.spelling:
            signature["calling_convention"] = "cdecl"
        
        return signature

    def _generate_overload_id(self, cursor) -> str:
        """Generate unique ID for function overloads considering parameters."""
        element_id = self.element_tracker.generate_element_id(cursor, "function")
        param_types = [arg.type.spelling for arg in cursor.get_arguments()]
        return f"{element_id}:{':'.join(param_types)}"

    def _process_class(self, cursor) -> None:
        """Process a class/struct declaration."""
        class_name = cursor.spelling
        self.log(f"_process_class:\t{cursor.spelling}\tcursor.kind = {cursor.kind}", "note")

        class_data = {
            "type": "class",
            "name": class_name,
            "declaration": self.range_locator.get_code_snippet(cursor),
            "methods": [],
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_class(class_data)

    def _process_namespace(self, cursor) -> None:
        """Process a namespace declaration."""
        namespace_name = cursor.spelling or "(anonymous)"
    
        namespace_data = {
            "type": "namespace",
            "name": namespace_name,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_namespace(namespace_data)

    def _process_class_template(self, cursor) -> None:
        """Process a class template declaration."""
        template_name = cursor.spelling
        if not template_name:  # For anonymous template classes
            template_name = f"anonymous_template_at_{cursor.location.line}"
        
        self.log(f"_process_class_template:\t{template_name}\tcursor.kind = {cursor.kind}", "note")

        # Get full template class definition
        code = self.range_locator.get_code_snippet(cursor)
        
        template_data = {
            "type": "class_template",
            "name": template_name,
            "code": code,
            "methods": [],
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "template_parameters": self._get_template_parameters(cursor)
        }
        
        self.data_storage.add_class_template(template_data)

    def _process_function(self, cursor) -> None:
        """Process a free function definition."""
        if not cursor.is_definition():
            return
        
        self.log(f"_process_function:\t{cursor.spelling}\tcursor.kind = {cursor.kind}", "note")

        code = self.range_locator.get_code_snippet(cursor)
        if not code:
            return

        function_info = {
            "type": "function",
            "name": cursor.spelling,
            "code": code,
            "signature": self._get_function_signature(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "is_overloaded": False,
            "overload_count": 1
        }

        self.data_storage.add_function(function_info)

    def _process_class_member(self, cursor) -> None:
        """Process a class/struct/template method."""
        parent = cursor.semantic_parent
        if not parent:
            return
        
        self.log(f"_process_class_member:\t{parent.spelling}::{cursor.spelling}\tparent.kind = {parent.kind}\tcursor.kind = {cursor.kind}", "note")
        is_regular_class = parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL)
        is_template = parent.kind in (
            CursorKind.CLASS_TEMPLATE, 
            CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION
        )
        
        if not (is_regular_class or is_template):
            return
        
        if is_regular_class and not cursor.is_definition():
            return
        
        parent_name = parent.spelling or f"anon_at_{parent.location.line}"
       
        # Get both declaration and full body for template methods
        code = self.range_locator.get_code_snippet(cursor)
        full_body = self.template_extractor.get_template_method_body(cursor) if is_template else None
        
        if not code and not full_body:
            return
        
        method_info = {
            "type": "method",
            "name": cursor.spelling,
            "code": code,
            "full_body": full_body if full_body else code,  # Use full body when available
            "signature": self._get_function_signature(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "is_template": is_template,
            "parent": parent_name,
            "is_overloaded": False,
            "overload_count": 1
        }
        
        if is_template:
            method_info["template_parameters"] = self._get_template_parameters(cursor)
            method_info["parent_template"] = parent_name
            
            if cursor.is_definition():
                method_info["inline_definition"] = True
            
            if parent_name not in self.data_storage.class_templates:
                self._process_class_template(parent)
            self.data_storage.class_templates[parent_name]["methods"].append(method_info)
        else:
            
            if parent_name not in self.data_storage.classes:
                self._process_class(parent)
            self.data_storage.classes[parent_name]["methods"].append(method_info)

    def _get_template_parameters(self, cursor) -> List[str]:
        """Extract template parameters from template declaration."""
        params = []
        for child in cursor.get_children():
            if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                params.append(child.spelling)
        return params
    
    def _process_template(self, cursor) -> None:
        """Process a function template."""
        if not (code := self.range_locator.get_code_snippet(cursor)):
            return
        
        # Check if this is a method template (inside a class/struct/template)
        parent = cursor.semantic_parent
        self.log(f"_process_template:\t{parent.spelling}::{cursor.spelling}\tparent.kind = {parent.kind}\tcursor.kind = {cursor.kind}", "note")
        
        # Get full body for template functions
        full_body = self.template_extractor.get_template_method_body(cursor)
        
        if parent and parent.kind in (
            CursorKind.CLASS_DECL, 
            CursorKind.STRUCT_DECL,
            CursorKind.CLASS_TEMPLATE,
            CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION
        ):
            parent_name = parent.spelling or f"anon_at_{parent.location.line}"
            
            method_info = {
                "type": "template_method",
                "name": cursor.spelling,
                "code": code,
                "full_body": full_body if full_body else code,
                "signature": self._get_function_signature(cursor),
                "location": self._get_relative_path(cursor.location.file.name),
                "line": cursor.location.line,
                "is_template": True,
                "parent": parent_name,
                "template_parameters": self._get_template_parameters(cursor)
            }
            
            if parent.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                method_info["parent_template"] = parent_name
                if parent_name not in self.data_storage.class_templates:
                    self._process_class_template(parent)
                self.data_storage.class_templates[parent_name]["methods"].append(method_info)
            else:
                if parent_name not in self.data_storage.classes:
                    self._process_class(parent)
                self.data_storage.classes[parent_name]["methods"].append(method_info)
        else:
            # Regular function template
            template_data = {
                "type": "template",
                "name": cursor.spelling,
                "code": code,
                "full_body": full_body if full_body else code,
                "location": self._get_relative_path(cursor.location.file.name),
                "line": cursor.location.line,
                "template_parameters": self._get_template_parameters(cursor),
                "signature": self._get_function_signature(cursor)
            }
            
            self.data_storage.add_function_template(template_data)

    def _process_lambda(self, cursor) -> None:
        """Process a lambda expression."""
        if not (code := self.range_locator.get_code_snippet(cursor)):
            return
            
        lambda_data = {
            "type": "lambda",
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_lambda(lambda_data)

    def _process_preprocessor_directive(self, cursor) -> None:
        """Process a preprocessor directive (#ifdef, #pragma, etc.)."""
        if not (code := self.range_locator.get_code_snippet(cursor)):
            return
            
        directive_data = {
            "type": "preprocessor",
            "directive": cursor.spelling,
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_preprocessor_directive(directive_data)

    def _process_error_handler(self, cursor) -> None:
        """Process a try-catch block."""
        if not (code := self.range_locator.get_code_snippet(cursor)):
            return
        
        handler_data = {
            "type": "error_handler",
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_error_handler(handler_data)

    def _process_macro(self, cursor) -> None:
        """Process a macro definition."""
        if not cursor.spelling:
            return

        macro_data = {
            "type": "macro",
            "name": cursor.spelling,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_macro(macro_data)

    def _process_literal(self, cursor) -> None:
        """Process a user-defined literal."""
        if not cursor.spelling:
            return
        
        literal_data = {
            "type": "literal",
            "name": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_literal(literal_data)

    def _process_attribute(self, cursor) -> None:
        """Process a C++ attribute ([[...]])."""
        if not cursor.spelling:
            return
        
        attribute_data = {
            "type": "attribute",
            "name": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_attribute(attribute_data)

    def _get_cursor_id(self, cursor) -> str:
        """Generate a unique ID for a cursor based on its location and kind."""
        if not cursor.location.file:
            return f"{cursor.kind}:{cursor.spelling}"
        
        file_path = self._get_relative_path(cursor.location.file.name)
        return f"{cursor.kind}:{file_path}:{cursor.location.line}:{cursor.location.column}:{cursor.spelling}"

    def visit_node(self, cursor):
        """Recursively visit AST nodes and process them."""
        if cursor.kind == CursorKind.TRANSLATION_UNIT:
            pass
        else:
            # Пропускаем системные заголовки и файлы из skip-листа
            if (self.file_processor.is_system_header(cursor.location.file.name if cursor.location.file else None) or
                (self.skip_files_func and self.skip_files_func(self._get_relative_path(cursor.location.file.name)))):
                return
            
            # Проверяем, был ли курсор уже обработан
            cursor_id = self._get_cursor_id(cursor)
            if self.element_tracker.is_processed(cursor_id):
                return
            
            # Обрабатываем только релевантные виды курсоров
            if cursor.kind in self._RELEVANT_CURSOR_KINDS:
                self.element_tracker.mark_processed(cursor_id)
                
                # Для классов/структур проверяем, что это определение
                if cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
                    if cursor.is_definition():
                        self._cursor_handlers[cursor.kind](cursor)
                else:
                    # Для остальных типов просто вызываем соответствующий обработчик
                    handler = self._cursor_handlers.get(cursor.kind)
                    if handler:
                        handler(cursor)
                    else:
                        self.element_tracker.track_unhandled_kind(cursor)
        
        # Рекурсивно обрабатываем дочерние узлы
        for child in cursor.get_children():
            self.visit_node(child)

    def process_file(self, file_path: Path) -> None:
        """Process a single source file."""
        translation_unit = self.file_processor.parse_file(file_path, self.skip_files_func)
        
        if not translation_unit:
            return      
        
        self.visit_node(translation_unit.cursor)

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all extracted code structures."""
        return self.data_storage.get_all_data()

    @property
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about unprocessed elements."""
        return self.element_tracker.unprocessed_stats
    

def main() -> None:
    Config.set_library_file(r"C:\\work\\clang-llvm-20.1.7-windows-msvc\\clang\\bin\\libclang.dll")
    BASE_ROOT = r"C:\\work\\llm_test"
    # PROJ_NAME = r"overload_example"
    # PROJ_NAME = r"template_exampl"
    PROJ_NAME = r"simple"
    REPO_PATH = os.path.join(BASE_ROOT, "codebase", PROJ_NAME)
    OUTPUT_JSONL = os.path.join(BASE_ROOT, f"dataset_clang_{PROJ_NAME}.jsonl")
    log_level = 1

    tango_generated_stop_list = [
        "main.cpp",
        "*Class.cpp",
        "*Class.h",
        "*DynAttrUtils.cpp",
        "*StateMachine.cpp",
        "ClassFactory.cpp",
    ]

    def should_skip_file(filename):
        for pattern in tango_generated_stop_list:
            if pattern.startswith('*'):
                if filename.endswith(pattern[1:]):
                    return True
            else:
                if filename == pattern:
                    return True
        return False
        
    # Initialize data storage with output path
    data_storage = JsonDataStorage(OUTPUT_JSONL)
    extractor = CodeExtractor(REPO_PATH, should_skip_file, data_storage=data_storage, log_level = log_level)

    file_count = 0
    for root, _, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx")):
                if should_skip_file(file):
                    print(f"Skipping excluded file: {file}")
                    continue

                file_path = Path(root) / file
                print(f"\nProcessing: {file_path}")
                try:
                    extractor.process_file(file_path)
                    file_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")
                    traceback.print_exc()

    
    # Print statistics and save results
    data_storage.print_statistics(extractor.unprocessed_stats)
    data_storage.save_to_file()
    
    print(f"Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()