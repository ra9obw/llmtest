import os
from pathlib import Path
import traceback 
from typing import List, Dict, Any, Optional, Callable, Set
from functools import partial
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TypeKind
from interfaces import IElementTracker, IFileProcessor
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
                 data_storage: Optional[JsonDataStorage] = None,
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
            CursorKind.FUNCTION_DECL: self._process_function,
            CursorKind.FUNCTION_TEMPLATE: self._process_function_template,
            CursorKind.LAMBDA_EXPR: self._process_lambda,
            CursorKind.PREPROCESSING_DIRECTIVE: self._process_preprocessor_directive,
            CursorKind.MACRO_DEFINITION: self._process_macro,
            CursorKind.CONVERSION_FUNCTION: self._process_literal,
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
        parent_type = parent.kind.name
        parent_name = parent.spelling or "(anonymous)"
        
        return parent_id, parent_type, parent_name

    def _process_class(self, cursor) -> None:
        if not cursor.is_definition():
            self.log(f"!!not cursor.is_definition():\t_process_class {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
            return

        self.log(f"{cursor.kind}:\t_process_class {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        class_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "class",
            "name": cursor.spelling,
            "kind": str(cursor.kind),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "column": cursor.location.column,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        
        self.data_storage.add_element("classes", class_data)
        return False

    def _process_class_template(self, cursor) -> None:
        self.log(f"{cursor.kind}:\t_process_class_template {cursor.spelling or f"anon_template_{cursor.location.line}"} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        template_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "class_template",
            "name": cursor.spelling or f"anon_template_{cursor.location.line}",
            "code": self.range_locator.get_code_snippet(cursor),
            "full_body": self.template_extractor.get_template_method_body(cursor),
            "template_parameters": self._get_template_parameters(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        
        self.data_storage.add_element("class_templates", template_data)
        return False

    def _process_method(self, cursor) -> None:
        parent = cursor.semantic_parent
        if not parent:
            self.log(f"{cursor.kind}:\t_process_method {cursor.spelling} HAS NO PARENT!", 2)
            return
        
        parent_type = parent.kind.name
        
        self.log(f"{cursor.kind}:\t_process_method {cursor.spelling} of {parent.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id = self.element_tracker.generate_element_id(parent)
        
        if parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            code = self.range_locator.get_code_snippet(cursor)
            is_defined = cursor.is_definition()
        else:
            code_definition = self.range_locator.get_code_snippet(cursor)
            body = self.template_extractor.get_template_method_body(cursor)
            if body:
                is_defined = True
                code = code_definition + "\n" + body
            else:
                is_defined = False
                code = code_definition

        method_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "method",
            "name": cursor.spelling,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent.spelling,
            "signature": self._get_function_signature(cursor),
            "code": code,
            "is_defined": str(is_defined),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "is_template": parent.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION)
        }
        
        self.data_storage.add_element("methods", method_data)
        return True

    def _process_function(self, cursor) -> None:
        if not cursor.is_definition():
            return
        self.log(f"{cursor.kind}:\t_process_method {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        parent_id, parent_type, parent_name = self._get_parent_info(cursor)
        is_defined = cursor.is_definition()

        function_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "function",
            "name": cursor.spelling,
            "signature": self._get_function_signature(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name,
            "is_defined": str(is_defined)
        }
        
        self.data_storage.add_element("functions", function_data)
        return True

    def _process_function_template(self, cursor) -> None:
        parent = cursor.semantic_parent
        
        if parent and parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            self.log(f"{cursor.kind.name}:\t_process_function_template {cursor.spelling} of {parent.kind} : {parent.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
            
            parent_id = self.element_tracker.generate_element_id(parent)
            if not parent_id:
                self.log(f"{cursor.kind.name}:\t_process_method {cursor.spelling} of {parent.spelling} NO PARENT FOUND!", 2)
                parent_id = "NO PARENT"
                parent_type = "no_parent"
            else:
                parent_type = "class" if parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) else "class_template"

            code_definition = self.range_locator.get_code_snippet(cursor)
            body = self.template_extractor.get_template_method_body(cursor)
            if body:
                is_defined = True
                code = code_definition + "\n" + body
            else:
                is_defined = False
                code = code_definition

            method_data = {
                "id": self.element_tracker.generate_element_id(cursor),
                "type": "method",
                "name": cursor.spelling,
                "parent_id": parent_id,
                "parent_type": parent_type,
                "parent_name": parent.spelling,
                "signature": self._get_function_signature(cursor),
                "code": code,
                "is_defined": str(is_defined),
                "location": self.file_processor.get_relative_path(cursor.location.file.name),
                "line": cursor.location.line,
                "is_template": parent.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION)
            }
            self.data_storage.add_element("methods", method_data)
        else:
            self.log(f"{cursor.kind}:\t_process_function_template {cursor.spelling} in {self.file_processor.get_relative_path(cursor.location.file.name)}")
            self.log(f"\t is_definition = {cursor.is_definition()}")

            parent_id, parent_type, parent_name = self._get_parent_info(cursor)

            code_definition = self.range_locator.get_code_snippet(cursor)
            body = self.template_extractor.get_template_method_body(cursor)
            if body:
                is_defined = True
                code = code_definition + "\n" + body
            else:
                is_defined = False
                code = code_definition

            template_data = {
                "id": self.element_tracker.generate_element_id(cursor),
                "type": "function_template",
                "name": cursor.spelling,
                "parent_id": parent_id,
                "parent_type": parent_type,
                "parent_name": parent_name,
                "signature": self._get_function_signature(cursor),
                "code": code,
                "is_defined": str(is_defined),
                "template_parameters": self._get_template_parameters(cursor),
                "location": self.file_processor.get_relative_path(cursor.location.file.name),
                "line": cursor.location.line,
            }
            
            self.data_storage.add_element("function_templates", template_data)
        return True

    def _process_namespace(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        _kind = cursor.kind.name
        _file_path = self.file_processor.get_relative_path(cursor.location.file.name).replace(":", "_")
        _name = cursor.spelling or f"namespace_{_kind}:{_file_path}:{cursor.location.line}:{cursor.location.column}"

        self.log(f"{_kind}:\t_process_namespace {cursor.spelling} of {parent_type} : {parent_name} in {self.file_processor.get_relative_path(cursor.location.file.name)}")

        namespace_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "namespace",
            "name": _name,
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        self.data_storage.add_element("namespaces", namespace_data)
        return False

    def _process_lambda(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        _kind = cursor.kind.name
        _file_path = self.file_processor.get_relative_path(cursor.location.file.name).replace(":", "_")
        _name = f"lambda_{_kind}:{_file_path}:{cursor.location.line}:{cursor.location.column}"

        lambda_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "lambda",
            "name": _name,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        self.data_storage.add_element("lambdas", lambda_data)
        return True

    def _process_preprocessor_directive(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        directive_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "preprocessor",
            "directive": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        self.data_storage.add_element("preprocessor_directives", directive_data)
        return True

    def _process_macro(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        macro_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "macro",
            "name": cursor.spelling,
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        self.data_storage.add_element("macros", macro_data)
        return True

    def _process_literal(self, cursor) -> None:
        parent_id, parent_type, parent_name = self._get_parent_info(cursor)

        literal_data = {
            "id": self.element_tracker.generate_element_id(cursor),
            "type": "literal",
            "name": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self.file_processor.get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "parent_id": parent_id,
            "parent_type": parent_type,
            "parent_name": parent_name
        }
        self.data_storage.add_element("literals", literal_data)
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
                print(cursor.kind.name, cursor.spelling)

            for child in cursor.get_children():
                self.visit_node(child)
        else:
            print(f"\tElement is processed:\t{cursor.kind.name}, {cursor.spelling}")

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
    PROJ_NAME = r"adc4x250"
    REPO_PATH = os.path.join(BASE_ROOT, "codebase", PROJ_NAME)
    OUTPUT_JSONL = os.path.join(BASE_ROOT, f"dataset_clang_{PROJ_NAME}.jsonl")

    tango_generated_stop_list = [
        "main.cpp",
        "*Class.cpp",
        "*Class.h",
        "*DynAttrUtils.cpp",
        "*StateMachine.cpp",
        "ClassFactory.cpp",
    ]

    def should_skip_file(filename):
        """Check if file should be skipped based on stop list"""
        for pattern in tango_generated_stop_list:
            if pattern.startswith('*'):
                # Pattern matches end of filename
                if filename.endswith(pattern[1:]):
                    return True
            else:
                # Exact match
                if filename == pattern:
                    return True
        return False

    data_storage = JsonDataStorage(OUTPUT_JSONL)
    extractor = CodeExtractor(REPO_PATH, data_storage=data_storage, skip_files_func=should_skip_file)

    for root, _, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx")):
                file_path = Path(root) / file
                print(f"Processing: {file_path}")
                try:
                    extractor.process_file(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    traceback.print_exc()

    data_storage.save_to_file()
    print(f"Results saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()