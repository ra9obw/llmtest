import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from clang.cindex import Cursor, CursorKind, TranslationUnit
from interfaces import IFileProcessor, IRangeLocator
from code_processing.code_cleaner import CodeCleaner

class RangeLocator(IRangeLocator):
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
        # print(f"_build_file_positions_cache {translation_unit.spelling}")
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
                    "end_line": node.extent.end.line,
                    "end_col": node.extent.end.column,
                    "file": node.location.file.name,
                    "kind": node.kind
                })
            
            for child in node.get_children():
                collect_positions(child, level + 1)
        
        collect_positions(translation_unit.cursor)
        for file_path, positions in positions_dict.items():
            # print(f"builded for: {file_path}, cursors =  {len(positions)} pcs")
            positions.sort(key=lambda x: (x["line"], x["column"]))
            self._file_positions_cache[file_path] = positions

    def get_sibling_and_parent_positions(self, cursor: Cursor) -> List[Dict]:
        """Get sorted list of all cursor positions in the same file."""
        if not cursor or not cursor.location.file:
            return []
            
        file_path = cursor.location.file.name
        # print(f"get_sibling_and_parent_positions {file_path}")
        
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
                "end_line": pos["end_line"],
                "end_col": pos["end_col"],
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
                "end_line": next_cursor["end_line"],
                "end_col": next_cursor["end_col"],
                "kind": next_cursor["kind"]
            }
            
        return None
    
    def get_previous_cursor_position(self, cursor: Cursor, verbose) -> Optional[Dict]:
        """Get the position of the previous cursor at the same or higher level before the current one."""
        siblings = self.get_sibling_and_parent_positions(cursor)
        if not siblings:
            return None
            
        current_index = next(
            (i for i, sibling in enumerate(siblings) 
            if sibling["is_current"]
            ), -1)
        
        if current_index == -1:
            return None
        if verbose:
            print(self._file_positions_cache[cursor.location.file.name])    
        # Ищем предыдущие позиции с уровнем <= текущему
        for i in range(current_index - 1, -1, -1):
            if siblings[i]["level"] <= siblings[current_index]["level"]:
                return {
                    "file": siblings[i]["file"],
                    "line": siblings[i]["line"],
                    "column": siblings[i]["column"],
                    "level": siblings[i]["level"],
                    "end_line": siblings[i]["end_line"],
                    "end_col": siblings[i]["end_col"],
                    "kind": siblings[i]["kind"]
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

    def get_context(self, cursor) -> Dict[str, List[str]]:
        """Extracts context around the cursor position.
        
        Returns:
            Dict with two keys:
            - "context_before": list of up to 3 lines before the cursor
            - "context_after": list of up to 3 lines starting from the cursor line
        """
        file_path = cursor.location.file.name
        if not file_path or not os.path.exists(file_path):
            return {"context_before": [], "context_after": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return {"context_before": [], "context_after": []}
        
        line_num = cursor.location.line - 1  # convert to 0-based index
        if line_num < 0 or line_num >= len(lines):
            return {"context_before": [], "context_after": []}
        
        # Get context before (up to 3 lines)
        start_before = max(0, line_num - 3)
        context_before = [line.strip() for line in lines[start_before:line_num]]
        
        # Get context after (up to 3 lines including current line)
        end_after = min(len(lines), line_num + 3)
        context_after = [line.strip() for line in lines[line_num:end_after]]
        
        return {
            "context_before": context_before,
            "context_after": context_after
        }
    
    def get_comments_in_range(self, file_path, start_line, end_line):
        """Extracts all comments in the specified line range."""
        comments = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            return comments
        
        in_block_comment = False
        current_comment = None
        
        for i, line in enumerate(lines, 1):
            if i < start_line:
                continue
            if i > end_line:
                break
                
            stripped = line.strip()
            
            # Handle block comments
            if '/*' in line and '*/' in line:
                comments.append({
                    "type": "block",
                    "text": line,
                    "line": i
                })
            elif '/*' in line:
                in_block_comment = True
                current_comment = {
                    "type": "block",
                    "text": line,
                    "line": i
                }
            elif '*/' in line and in_block_comment:
                in_block_comment = False
                current_comment["text"] += line
                comments.append(current_comment)
                current_comment = None
            elif in_block_comment:
                current_comment["text"] += line
            elif stripped.startswith('//'):
                comments.append({
                    "type": "line",
                    "text": line,
                    "line": i
                })
        
        return comments

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