from pathlib import Path
from typing import Dict, List, Optional, Set
from clang.cindex import Cursor, CursorKind, TranslationUnit
from interfaces import IFileProcessor
from code_processing.code_cleaner import CodeCleaner

class RangeLocator:
    """Класс для работы с позициями в коде и определения границ элементов."""
    
    def __init__(self, cursor_of_interst: Set[CursorKind], file_processor: IFileProcessor):
        self._file_positions_cache: Dict[str, List[Dict]] = {}
        self._RELEVANT_CURSOR_KINDS = cursor_of_interst
        self.file_processor = file_processor
        self.code_cleaner = CodeCleaner()
    
    def _build_file_positions_cache(self, translation_unit: TranslationUnit) -> None:
        """Build position cache for all relevant cursors in the file."""
        # print(f"_build_file_positions_cache {translation_unit.get_file()}")
        if not translation_unit.cursor:
            print("not translation_unit.cursor")
            return
            
        # file_path = translation_unit.spelling
        # if not file_path or file_path in self._file_positions_cache:
            # print("not file_path")
            # return
        print(f"_build_file_positions_cache {translation_unit.spelling}")
        positions_dict = {}
        
        def collect_positions(node, level=0):
            
            if (node.location.file 
                and not self.file_processor.is_system_header(node.location.file.name) 
                and node.kind in self._RELEVANT_CURSOR_KINDS):
                if node.location.file.name not in positions_dict.keys():
                    positions_dict[node.location.file.name] = []
                positions_dict[node.location.file.name].append({
                    "cursor": node,
                    "level": level,
                    "line": node.location.line,
                    "column": node.location.column,
                    "file": node.location.file.name,
                    "kind": node.kind
                })
            
            for child in node.get_children():
                collect_positions(child, level + 1)
        
        collect_positions(translation_unit.cursor)
        for file_path, positions in positions_dict.items():
            print(f"builded for: {file_path}, cursors =  {len(positions)} pcs")
            positions.sort(key=lambda x: (x["line"], x["column"]))
            self._file_positions_cache[file_path] = positions

    def get_sibling_and_parent_positions(self, cursor: Cursor) -> List[Dict]:
        """Get sorted list of all cursor positions in the same file."""
        if not cursor or not cursor.location.file:
            return []
            
        file_path = cursor.location.file.name
        print(f"get_sibling_and_parent_positions {file_path}")
        
        if file_path not in self._file_positions_cache:
            self._build_file_positions_cache(cursor.translation_unit)
            
        cached_positions = self._file_positions_cache.get(file_path, [])
        if not cached_positions:
            return []
        
        current_level = None
        current_index = -1
        
        for i, pos in enumerate(cached_positions):
            if pos["cursor"] == cursor:
                current_level = pos["level"]
                current_index = i
                break
                
        if current_level is None:
            return []
            
        filtered_positions = [
            pos for pos in cached_positions 
            if pos["level"] <= current_level
        ]
        
        current_in_filtered = next(
            (i for i, pos in enumerate(filtered_positions) 
            if pos["cursor"] == cursor
        ), -1)
        
        result = []
        for pos in filtered_positions:
            result.append({
                "file": pos["file"],
                "line": pos["line"],
                "column": pos["column"],
                "level": pos["level"],
                "kind": str(pos["kind"]),
                "is_current": pos["cursor"] == cursor
            })
            
        return result

    def get_next_cursor_position(self, cursor: Cursor) -> Optional[Dict]:
        """Get the position of the next cursor after the current one."""
        siblings = self.get_sibling_and_parent_positions(cursor)
        if not siblings:
            return None
            
        current_index = next(
            (i for i, sibling in enumerate(siblings) 
            if sibling["is_current"]
        ), -1)
        
        if current_index == -1:
            return None
            
        if current_index + 1 < len(siblings):
            next_cursor = siblings[current_index + 1]
            return {
                "file": next_cursor["file"],
                "line": next_cursor["line"],
                "column": next_cursor["column"],
                "level": next_cursor["level"],
                "kind": next_cursor["kind"]
            }
            
        return None

    def get_code_snippet(self, cursor: Cursor) -> Optional[str]:
        """Extract code snippet for the given cursor with line and column precision."""
        if not cursor.location or not cursor.location.file:
            return None
            
        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            start_line = cursor.extent.start.line - 1
            start_col = cursor.extent.start.column - 1
            end_line = cursor.extent.end.line - 1
            end_col = cursor.extent.end.column
            
            if start_line == end_line:
                line = lines[start_line]
                code = line[start_col:end_col]
            else:
                code = [lines[start_line][start_col:]]
                for line_num in range(start_line + 1, end_line):
                    code.append(lines[line_num])
                if end_line < len(lines):
                    code.append(lines[end_line][:end_col])
                code = ''.join(code)
            
            return self.code_cleaner.clean_code(code)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {cursor.location.file.name}: {e}")
            return None

    @staticmethod
    def get_offset_from_position(file_path: str, line: int, column: int) -> int:
        """Convert file position (line, column) to file offset."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.readlines()
            
            line = max(0, line - 1)
            if line >= len(content):
                return sum(len(line) for line in content)
            
            offset = sum(len(content[i]) for i in range(line))
            column = max(0, column - 1)
            offset += min(column, len(content[line]))
            
            return offset
        except Exception as e:
            print(f"[ERROR] Failed to convert position to offset: {e}")
            return 0

    @staticmethod
    def get_offset(location) -> int:
        """Helper function to convert cursor location to file offset"""
        if not location.file:
            return 0
            
        try:
            with open(location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            offset = 0
            for i in range(location.line - 1):
                offset += len(lines[i])
            
            offset += location.column - 1
            return offset
        except:
            return 0