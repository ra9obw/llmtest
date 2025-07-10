import os
from pathlib import Path
import traceback 
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
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
        
        self.element_tracker = element_tracker or ElementTracker()
        self.file_processor = file_processor or FileProcessor(self.repo_path)
        self.data_storage = data_storage or JsonDataStorage()

        self.log_level = log_level
        self._cursor_handlers = self._init_cursor_handlers()
        self._RELEVANT_CURSOR_KINDS = set(self._cursor_handlers.keys())

        self.range_locator = RangeLocator(self._RELEVANT_CURSOR_KINDS, self.file_processor)
        self.template_extractor = TemplateBodyExtractor(self.range_locator)

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
            CursorKind.CXX_TRY_STMT: self._process_error_handler,
            CursorKind.MACRO_DEFINITION: self._process_macro,
            CursorKind.CONVERSION_FUNCTION: self._process_literal,
            CursorKind.ANNOTATE_ATTR: self._process_attribute,
        }

    def _generate_id(self, cursor=None, element_type=None) -> str:
        """Generate unique ID for elements."""
        if cursor and element_type:
            file_path = self._get_relative_path(cursor.location.file.name) if cursor.location.file else "unknown"
            unique_str = f"{file_path}:{cursor.location.line}:{cursor.location.column}:{element_type}:{cursor.spelling}"
            return f"{element_type[:3]}_{hash(unique_str) & 0xFFFFFFFF}"
        return f"id_{uuid.uuid4().hex}"

    def _get_relative_path(self, absolute_path: str) -> str:
        try:
            return str(Path(absolute_path).resolve().relative_to(self.repo_path))
        except ValueError:
            return absolute_path

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

    def _process_class(self, cursor) -> None:
        if not cursor.is_definition():
            return

        class_data = {
            "id": self._generate_id(cursor, "class"),
            "type": "class",
            "name": cursor.spelling,
            "kind": str(cursor.kind),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "column": cursor.location.column
        }
        
        self.data_storage.add_element("classes", class_data)

    def _process_class_template(self, cursor) -> None:
        template_data = {
            "id": self._generate_id(cursor, "class_template"),
            "type": "class_template",
            "name": cursor.spelling or f"anon_template_{cursor.location.line}",
            "code": self.range_locator.get_code_snippet(cursor),
            "template_parameters": self._get_template_parameters(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("class_templates", template_data)

    def _process_method(self, cursor) -> None:
        parent = cursor.semantic_parent
        if not parent:
            return

        parent_id = self._get_parent_id(parent)
        if not parent_id:
            return

        method_data = {
            "id": self._generate_id(cursor, "method"),
            "type": "method",
            "name": cursor.spelling,
            "parent_id": parent_id,
            "parent_type": "class" if parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) else "class_template",
            "signature": self._get_function_signature(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "is_template": parent.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION)
        }
        
        self.data_storage.add_element("methods", method_data)

    def _get_parent_id(self, parent_cursor) -> Optional[str]:
        """Get or create ID for parent element."""
        if parent_cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            return self.data_storage.get_or_create_id("classes", {
                "name": parent_cursor.spelling,
                "location": self._get_relative_path(parent_cursor.location.file.name),
                "line": parent_cursor.location.line
            })
        elif parent_cursor.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            return self.data_storage.get_or_create_id("class_templates", {
                "name": parent_cursor.spelling,
                "location": self._get_relative_path(parent_cursor.location.file.name),
                "line": parent_cursor.location.line
            })
        return None

    def _process_function(self, cursor) -> None:
        if not cursor.is_definition():
            return

        function_data = {
            "id": self._generate_id(cursor, "function"),
            "type": "function",
            "name": cursor.spelling,
            "signature": self._get_function_signature(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("functions", function_data)

    def _process_function_template(self, cursor) -> None:
        template_data = {
            "id": self._generate_id(cursor, "function_template"),
            "type": "function_template",
            "name": cursor.spelling,
            "signature": self._get_function_signature(cursor),
            "code": self.range_locator.get_code_snippet(cursor),
            "template_parameters": self._get_template_parameters(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("function_templates", template_data)

    def _process_namespace(self, cursor) -> None:
        namespace_data = {
            "id": self._generate_id(cursor, "namespace"),
            "type": "namespace",
            "name": cursor.spelling or "(anonymous)",
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("namespaces", namespace_data)

    def _process_lambda(self, cursor) -> None:
        lambda_data = {
            "id": self._generate_id(cursor, "lambda"),
            "type": "lambda",
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("lambdas", lambda_data)

    def _process_preprocessor_directive(self, cursor) -> None:
        directive_data = {
            "id": self._generate_id(cursor, "preprocessor"),
            "type": "preprocessor",
            "directive": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("preprocessor_directives", directive_data)

    def _process_error_handler(self, cursor) -> None:
        handler_data = {
            "id": self._generate_id(cursor, "error_handler"),
            "type": "error_handler",
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("error_handlers", handler_data)

    def _process_macro(self, cursor) -> None:
        macro_data = {
            "id": self._generate_id(cursor, "macro"),
            "type": "macro",
            "name": cursor.spelling,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("macros", macro_data)

    def _process_literal(self, cursor) -> None:
        literal_data = {
            "id": self._generate_id(cursor, "literal"),
            "type": "literal",
            "name": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("literals", literal_data)

    def _process_attribute(self, cursor) -> None:
        attribute_data = {
            "id": self._generate_id(cursor, "attribute"),
            "type": "attribute",
            "name": cursor.spelling,
            "code": self.range_locator.get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        self.data_storage.add_element("attributes", attribute_data)

    def _get_template_parameters(self, cursor) -> List[str]:
        params = []
        for child in cursor.get_children():
            if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                params.append(child.spelling)
        return params

    def _get_cursor_id(self, cursor) -> str:
        if not cursor.location.file:
            return f"{cursor.kind}:{cursor.spelling}"
        
        file_path = self._get_relative_path(cursor.location.file.name)
        return f"{cursor.kind}:{file_path}:{cursor.location.line}:{cursor.location.column}:{cursor.spelling}"

    def visit_node(self, cursor):
        if cursor.kind == CursorKind.TRANSLATION_UNIT:
            for child in cursor.get_children():
                self.visit_node(child)
            return

        if self._should_skip_cursor(cursor):
            return

        cursor_id = self._get_cursor_id(cursor)
        if self.element_tracker.is_processed(cursor_id):
            return

        if cursor.kind in self._RELEVANT_CURSOR_KINDS:
            self.element_tracker.mark_processed(cursor_id)
            handler = self._cursor_handlers.get(cursor.kind)
            if handler:
                handler(cursor)

        for child in cursor.get_children():
            self.visit_node(child)

    def _should_skip_cursor(self, cursor) -> bool:
        file_path = cursor.location.file.name if cursor.location.file else None
        if not file_path:
            return True
        return (self.file_processor.is_system_header(file_path) or \
               (self.skip_files_func and self.skip_files_func(self._get_relative_path(file_path))))

    def process_file(self, file_path: Path) -> None:
        translation_unit = self.file_processor.parse_file(file_path, self.skip_files_func)
        if translation_unit:
            self.visit_node(translation_unit.cursor)

    @property
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        return self.element_tracker.unprocessed_stats

def main() -> None:
    Config.set_library_file(r"C:\\work\\clang-llvm-20.1.7-windows-msvc\\clang\\bin\\libclang.dll")
    BASE_ROOT = r"C:\\work\\llm_test"
    PROJ_NAME = r"simple"
    REPO_PATH = os.path.join(BASE_ROOT, "codebase", PROJ_NAME)
    OUTPUT_JSONL = os.path.join(BASE_ROOT, f"dataset_clang_{PROJ_NAME}.jsonl")
    
    data_storage = JsonDataStorage(OUTPUT_JSONL)
    extractor = CodeExtractor(REPO_PATH, data_storage=data_storage)

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