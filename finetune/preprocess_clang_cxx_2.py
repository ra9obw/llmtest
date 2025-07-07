import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from clang.cindex import Index, CursorKind, Config, TranslationUnit


class CodeExtractor:
    """Extracts class declarations and methods from C++ source files, ignoring system headers."""
    
    def __init__(self, repo_path: str) -> None:
        """Initialize the code extractor with repository path."""
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self.include_dirs = self._find_include_dirs()
        
        # Data storage
        self.classes: Dict[str, Dict] = {}
        
        # Track all processed elements to avoid duplicates
        self._processed_elements = set()

    def _find_include_dirs(self) -> List[str]:
        """Find all include directories in the repository."""
        include_dirs = [str(self.repo_path)]
        for root, dirs, _ in os.walk(self.repo_path):
            if 'include' in dirs:
                include_dirs.append(str(Path(root) / 'include'))
        return include_dirs

    def _get_compiler_args(self) -> List[str]:
        """Get compiler arguments for clang parsing."""
        args = [
            '-std=c++17',
            '-x', 'c++',
            '-fparse-all-comments',
            '-D__clang__'
        ]
        args.extend(arg for include_dir in self.include_dirs 
                   for arg in ['-I', include_dir])
        return args

    def _is_system_header(self, file_path: Optional[str]) -> bool:
        """Check if the file is a system header."""
        if not file_path:
            return True
        _is_system = file_path.startswith('/usr/include') or \
               file_path.startswith('/usr/local/include') or \
               file_path.startswith('/Applications/Xcode.app/') or \
               file_path.startswith('C:\\Program Files (x86)\\Microsoft Visual Studio') or \
               file_path.startswith('C:\\Program Files\\Microsoft Visual Studio') or \
               '\\Windows Kits\\' in file_path or \
               file_path.startswith('<')
        if _is_system:
            print("system include found: {file_path}")
        return _is_system

    def _get_relative_path(self, absolute_path: str) -> str:
        """Convert absolute path to relative path from repo root."""
        try:
            return str(Path(absolute_path).resolve().relative_to(self.repo_path))
        except ValueError:
            return absolute_path

    def _clean_code(self, code: str) -> str:
        """Clean code while preserving meaningful indentation."""
        if not code:
            return code

        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        
        while lines and not lines[0].strip():
            lines.pop(0)
        
        while lines and not lines[-1].strip():
            lines.pop()
        
        lines = [line.replace('\t', '    ') for line in lines]
        code = '\n'.join(lines)
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        return code

    def _get_code_snippet(self, cursor) -> Optional[str]:
        """Extract code snippet for the given cursor, ignoring system headers."""
        if not cursor.location.file or self._is_system_header(cursor.location.file.name):
            return None
            
        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            start_line = cursor.extent.start.line - 1
            end_line = cursor.extent.end.line
            code = ''.join(lines[start_line:end_line])
            return self._clean_code(code)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {cursor.location.file}: {e}")
            return None

    def _get_element_id(self, cursor, element_type: str) -> str:
        """Generate unique ID for an element to detect duplicates."""
        if self._is_system_header(cursor.location.file.name if cursor.location.file else None):
            return ""
            
        location = self._get_relative_path(cursor.location.file.name)
        return f"{element_type}:{location}:{cursor.location.line}:{cursor.spelling}"

    def _process_class(self, cursor) -> None:
        """Process a class/struct declaration, ignoring system headers."""
        if self._is_system_header(cursor.location.file.name if cursor.location.file else None):
            return
            
        class_name = cursor.spelling
        element_id = self._get_element_id(cursor, "class")
        
        if not element_id or self._is_processed(element_id) or class_name in self.classes:
            return
            
        self._mark_processed(element_id)
        
        self.classes[class_name] = {
            "type": "class",
            "name": class_name,
            "declaration": self._get_code_snippet(cursor),
            "methods": [],
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }

    def _process_method_or_function(self, cursor) -> None:
        """Process a method or free function definition, ignoring system headers."""
        if self._is_system_header(cursor.location.file.name if cursor.location.file else None):
            return
            
        code = self._get_code_snippet(cursor)
        if not code:
            return
            
        element_type = "method" if cursor.kind == CursorKind.CXX_METHOD else "function"
        element_id = self._get_element_id(cursor, element_type)
        
        if not element_id or self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        item = {
            "type": element_type,
            "name": cursor.spelling,
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        }
        
        parent = cursor.semantic_parent
        if parent and parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            class_name = parent.spelling
            if class_name not in self.classes:
                self._process_class(parent)
            self.classes[class_name]["methods"].append(item)

    def process_file(self, file_path: Path) -> None:
        """Process a single source file."""
        if self._is_system_header(str(file_path)):
            return
            
        translation_unit = self.index.parse(
            str(file_path),
            args=self._get_compiler_args()
        )
        
        if not translation_unit:
            return
            
        def visit_node(cursor):
            print(cursor.kind, cursor.location.file.name)
            """Recursively visit AST nodes and process them."""
            # Skip system headers
            if self._is_system_header(cursor.location.file.name if cursor.location.file else None):
                return
                
            # Class/Struct declarations
            if (cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) 
                    and cursor.is_definition()):
                self._process_class(cursor)
            
            # Methods and functions
            elif (cursor.kind in (CursorKind.CXX_METHOD, CursorKind.FUNCTION_DECL) 
                    and cursor.is_definition()):
                self._process_method_or_function(cursor)
            
            # Process children
            for child in cursor.get_children():
                visit_node(child)
        
        visit_node(translation_unit.cursor)

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all extracted classes with their methods."""
        return [cls for cls in self.classes.values() if cls["methods"]]


def main() -> None:
    # Укажите путь к libclang.dll
    Config.set_library_file(r"C:\\work\\clang-llvm-20.1.7-windows-msvc\\clang\\bin\\libclang.dll")
    # Настройки путей
    BASE_ROOT = r"C:\\work\\llm_test"
    # PROJ_NAME = r"adc4x250"
    PROJ_NAME = r"simple"
    REPO_PATH = os.path.join(BASE_ROOT, "codebase", PROJ_NAME)
    # REPO_PATH = r"C:\\work\\pavlenko\\llm_test\\codebase\\simple"
    
    # OUTPUT_JSONL = r"C:\\work\\pavlenko\\llm_test\\dataset_clang_test.jsonl"
    OUTPUT_JSONL = os.path.join(BASE_ROOT, f"dataset_clang_{PROJ_NAME}.jsonl")

    extractor = CodeExtractor(REPO_PATH)

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
    
    print(f"\nProcessed {file_count} files")
    print(f"Found {len(extractor.classes)} classes")
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in extractor.get_results():
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()