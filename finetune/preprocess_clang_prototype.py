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
        self.sys_includes_visited = 0
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
        print(f"include_dirs: len = {len(include_dirs)}, {include_dirs}")
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
            self.sys_includes_visited += 1
        return _is_system
    
    def _get_relative_path(self, absolute_path: str) -> str:
        """Convert absolute path to relative path from repo root."""
        try:
            return str(Path(absolute_path).resolve().relative_to(self.repo_path))
        except ValueError:
            return absolute_path

    def _get_element_id(self, cursor, element_type: str) -> str:
        """Generate unique ID for an element to detect duplicates."""
        location = self._get_relative_path(cursor.location.file.name)
        return f"{element_type}:{location}:{cursor.location.line}:{cursor.spelling}"

    def _is_processed(self, element_id: str) -> bool:
        """Check if element was already processed."""
        return element_id in self._processed_elements

    def _mark_processed(self, element_id: str) -> None:
        """Mark element as processed."""
        self._processed_elements.add(element_id)

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
        """Extract code snippet for the given cursor with line and column precision."""
        if not cursor.location.file:
            return None
            
        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            start_line = cursor.extent.start.line - 1  # Переводим в 0-based индекс
            start_col = cursor.extent.start.column - 1
            end_line = cursor.extent.end.line - 1
            end_col = cursor.extent.end.column
            
            # Для однострочных элементов
            if start_line == end_line:
                line = lines[start_line]
                code = line[start_col:end_col]
            else:
                # Первая строка
                code = [lines[start_line][start_col:]]
                
                # Промежуточные строки (если есть)
                for line_num in range(start_line + 1, end_line):
                    code.append(lines[line_num])
                
                # Последняя строка
                if end_line < len(lines):
                    code.append(lines[end_line][:end_col])
                
                code = ''.join(code)
            
            return self._clean_code(code)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {cursor.location.file}: {e}")
            return None
        
    def _find_next_sibling(self, cursor):
        """
        Find the position of the next sibling cursor at the same hierarchy level.
        If no siblings exist at this level, move up one level and search again.
        Continues until a sibling is found or hierarchy is exhausted.
        
        Args:
            cursor: The current cursor to find next sibling for
            
        Returns:
            Optional[Dict]: Information about next sibling or None if not found
        """
        if not cursor:
            return None
        
        current = cursor
        print(f"current sibling\t{current.kind} {current.location}")
        while current:
            # Try to find next sibling at current level
            parent = current.semantic_parent
            print(f"parent sibling\t{parent.kind} {parent.location}")
            if not parent:
                break
                
            found_current = False
            for sibling in parent.get_children():
                if self._is_system_header(sibling.location.file.name if sibling.location.file else None):
                    continue
                print(f"sibling\t{sibling.kind} {sibling.location}")
                # Skip until we find our current node
                if not found_current:
                    if sibling == current:
                        print("sibling == current:")
                        found_current = True
                        continue
                    continue
                    
                # Return first sibling after current
                print("CCCCCCCCC")
                return {
                    "file": sibling.location.file.name if sibling.location.file else None,
                    "line": sibling.location.line,
                    "column": sibling.location.column,
                    "kind": str(sibling.kind),
                    "spelling": sibling.spelling,
                    "code": self._get_code_snippet(sibling)
                }
            
            # If no siblings found, move up one level
            current = parent
        print("BBBBBBBBBBBB")
        return None
    
    def get_sibling_and_parent_positions(self, cursor):
        """
        Get sorted list of all cursor positions in the same file that are
        at the same level or higher in hierarchy compared to current cursor.
        
        Args:
            cursor: The reference cursor to compare against
            
        Returns:
            List[Dict]: Sorted list of cursor positions with hierarchy info
        """
        if not cursor or not cursor.location.file:
            return []
        
        file_path = cursor.location.file.name
        current_file_cursors = []
        
        # Get translation unit for the file
        translation_unit = cursor.translation_unit
        
        # Collect all cursors in the file with their hierarchy level
        def collect_cursors(node, level=0):
            if node.location.file and node.location.file.name == file_path:
                current_file_cursors.append({
                    "cursor": node,
                    "level": level,
                    "line": node.location.line,
                    "column": node.location.column,
                    "file": node.location.file.name
                })
            
            for child in node.get_children():
                collect_cursors(child, level + 1)
        
        collect_cursors(translation_unit.cursor)
        
        # Filter cursors that are at same level or higher than our cursor
        # First find our cursor's level
        current_level = None
        for c in current_file_cursors:
            if c["cursor"] == cursor:
                current_level = c["level"]
                break
        
        if current_level is None:
            return []
        
        # Get all cursors at same or higher level (lower or equal level number)
        filtered_cursors = [
            c for c in current_file_cursors 
            if c["level"] <= current_level
        ]
        
        # Sort by position in file (line, column)
        filtered_cursors.sort(key=lambda x: (x["line"], x["column"]))
        
        # Prepare result with relevant information
        result = []
        for c in filtered_cursors:
            result.append({
                "file": c["file"],
                "line": c["line"],
                "column": c["column"],
                "level": c["level"],
                "kind": str(c["cursor"].kind),
                "is_current": c["cursor"] == cursor
            })
        
        return result
    
    def get_next_cursor_position(self, cursor):
        """
        Get the position of the next cursor after the current one in the file,
        considering only cursors at the same or higher hierarchy level.
        Returns None if current cursor is the last one.
        
        Args:
            cursor: The reference cursor to compare against
            
        Returns:
            Optional[Dict]: Position info of next cursor or None if not found
        """
        siblings = self.get_sibling_and_parent_positions(cursor)
        
        if not siblings:
            return None
        
        # Find current cursor in the list
        current_index = None
        for i, sibling in enumerate(siblings):
            if sibling["is_current"]:
                current_index = i
                break
        
        if current_index is None:
            return None
        
        # Check if there's a next element
        if current_index + 1 < len(siblings):
            next_cursor = siblings[current_index + 1]
            return {
                "file":   next_cursor["file"],
                "line":   next_cursor["line"],
                "column": next_cursor["column"],
                "level":  next_cursor["level"],
                "kind":   next_cursor["kind"]
            }
        
        return None
        
    def _get_template_method_body(self, cursor) -> Optional[str]:
        """
        Extracts the complete method body (with curly braces) for template and non-template methods.
        Uses get_next_cursor_position to limit the search range for better performance and accuracy.
        If next cursor is None, searches until end of file.
        """
        if not cursor.location.file:
            return None

        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Convert to 0-based offsets
            start_pos = self._get_offset(cursor.extent.start)
            end_pos = self._get_offset(cursor.extent.end)
            
            # Get next cursor position to limit search range
            next_cursor_info = self.get_next_cursor_position(cursor)
            search_end = len(content) if next_cursor_info is None else self._get_offset_from_position(
                next_cursor_info['file'], 
                next_cursor_info['line'], 
                next_cursor_info['column']
            )
            
            # Find the opening brace after the function declaration
            brace_pos = content.find('{', end_pos, search_end)
            if brace_pos == -1:
                return None

            # Now find the matching closing brace within the limited range
            brace_count = 1
            current_pos = brace_pos + 1
            end_brace_pos = -1

            while current_pos < search_end and brace_count > 0:
                char = content[current_pos]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_brace_pos = current_pos
                        break
                current_pos += 1

            if end_brace_pos == -1:
                return None

            # Extract the complete method body
            method_body = content[brace_pos:end_brace_pos+1]
            return self._clean_code(method_body)

        except Exception as e:
            print(f"[ERROR] Failed to extract method body: {e}")
            return None

    def _get_offset_from_position(self, file_path: str, line: int, column: int) -> int:
        """
        Helper method to convert file position (line, column) to file offset.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.readlines()
            
            # Convert to 0-based line number
            line = max(0, line - 1)
            if line >= len(content):
                return sum(len(line) for line in content)
            
            # Sum lengths of previous lines
            offset = sum(len(content[i]) for i in range(line))
            
            # Add column position (convert to 0-based)
            column = max(0, column - 1)
            offset += min(column, len(content[line]))
            
            return offset
        except Exception as e:
            print(f"[ERROR] Failed to convert position to offset: {e}")
            return 0

    def _get_offset(self, location) -> int:
        """Helper function to convert cursor location to file offset"""
        if not location.file:
            return 0
            
        try:
            with open(location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Calculate offset up to the line before our target
            offset = 0
            for i in range(location.line - 1):
                offset += len(lines[i])
            
            # Add the column position (0-based)
            offset += location.column - 1
            return offset
        except:
            return 0
            
    def process_file(self, file_path: Path) -> None:
        """Process a single source file."""
        # if self._is_system_header(str(file_path)):
        #     return
            
        translation_unit = self.index.parse(
            str(file_path),
            args=self._get_compiler_args()
        )
        
        if not translation_unit:
            return
            
        def visit_node(cursor, visits):
            if cursor.kind == CursorKind.TRANSLATION_UNIT:
                pass
            else:
                # # Skip system headers
                if self._is_system_header(cursor.location.file.name if cursor.location.file else None):
                    return
                print(f"{visits}{visits*' '} kind = {cursor.kind}\tspelling = {cursor.spelling}\t def:\t{cursor.is_definition()}\tfile: {self._get_relative_path(cursor.location.file.name)} line: {cursor.location.line}, col: {cursor.location.column}")
                # print(self._get_code_snippet(cursor))
                visits += 1
                                # Templates
                if cursor.kind == CursorKind.FUNCTION_TEMPLATE:
                    print(f"template function parrent: {cursor.semantic_parent.spelling}")
                    # print(f"next sibling: {self._find_next_sibling(cursor)}")
                    pos = self.get_next_cursor_position(cursor)
                    print(f"Line {pos['line']}:{pos['column']} | "
                            f"Level {pos['level']} | "
                            f"Kind {pos['kind']}")
                    print("function code:\t", self._get_template_method_body(cursor))
                
            # # Class/Struct declarations
            # if (cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) 
            #         and cursor.is_definition()):
            #     self._process_class(cursor)
            
            # # Methods and functions
            # elif (cursor.kind in (CursorKind.CXX_METHOD, CursorKind.FUNCTION_DECL) 
            #         and cursor.is_definition()):
            #     self._process_method_or_function(cursor)
            
            """Recursively visit AST nodes and process them."""
            # Process children
            for child in cursor.get_children():
                visit_node(child, visits)
            visits -= 1
        
        visit_node(translation_unit.cursor, 0)

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all extracted classes with their methods."""
        return [cls for cls in self.classes.values() if cls["methods"]]


def main() -> None:
    # Укажите путь к libclang.dll
    Config.set_library_file(r"C:\\work\\pavlenko\\clang-llvm-windows-msvc-20-1-5\\bin\\libclang.dll")
    # Настройки путей
    REPO_PATH = r"C:\\work\\pavlenko\\llm_test\\codebase\\simple"
    OUTPUT_JSONL = r"C:\\work\\pavlenko\\llm_test\\dataset_clang_test.jsonl"
    
    extractor = CodeExtractor(REPO_PATH)
    
    file_count = 0
    for root, _, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx")):
                file_path = Path(root) / file
                print(f"\nProcessing: {file_path}")
                # try:
                if 1:
                    extractor.process_file(file_path)
                    file_count += 1
                # except Exception as e:
                #     print(f"[ERROR] Failed to process {file_path}: {e}")
    
    print(f"\nProcessed {file_count} files")
    print(f"Found {len(extractor.classes)} classes")
    print(f"sys_includes_visited =  {extractor.sys_includes_visited}")
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in extractor.get_results():
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()