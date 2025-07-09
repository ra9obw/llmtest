from typing import Optional
from clang.cindex import Cursor
from .range_locator import RangeLocator

class TemplateBodyExtractor:
    """Класс для извлечения тел шаблонных функций и методов."""
    
    def __init__(self, range_locator: RangeLocator):
        self.range_locator = range_locator
    
    def get_template_method_body(self, cursor: Cursor) -> Optional[str]:
        """Extracts the complete method body (with curly braces) for template methods."""
        if not cursor.location.file:
            return None

        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            start_pos = self.range_locator.get_offset(cursor.extent.start)
            end_pos = self.range_locator.get_offset(cursor.extent.end)
            
            next_cursor_info = self.range_locator.get_next_cursor_position(cursor)
            search_end = len(content) if next_cursor_info is None else self.range_locator.get_offset_from_position(
                next_cursor_info['file'], 
                next_cursor_info['line'], 
                next_cursor_info['column']
            )

            # Find the opening brace after the function declaration
            brace_pos = end_pos
            in_line_comment = False
            in_block_comment = False
            in_string = False
            
            while brace_pos < search_end:
                char = content[brace_pos]
                next_char = content[brace_pos+1] if brace_pos+1 < len(content) else None
                
                if not in_string and not in_block_comment and char == '/' and next_char == '/':
                    in_line_comment = True
                elif not in_string and not in_line_comment and char == '/' and next_char == '*':
                    in_block_comment = True
                elif in_block_comment and char == '*' and next_char == '/':
                    in_block_comment = False
                    brace_pos += 1
                elif char == '\n':
                    in_line_comment = False
                elif char == '"' or char == "'":
                    if not in_line_comment and not in_block_comment:
                        in_string = not in_string
                
                if not in_line_comment and not in_block_comment and not in_string and char == '{':
                    break
                    
                brace_pos += 1
            else:
                return None

            # Find the matching closing brace
            brace_count = 1
            current_pos = brace_pos + 1
            end_brace_pos = -1
            
            in_line_comment = False
            in_block_comment = False
            in_string = False
            
            while current_pos < search_end and brace_count > 0:
                char = content[current_pos]
                next_char = content[current_pos+1] if current_pos+1 < len(content) else None
                
                if not in_string and not in_block_comment and char == '/' and next_char == '/':
                    in_line_comment = True
                elif not in_string and not in_line_comment and char == '/' and next_char == '*':
                    in_block_comment = True
                elif in_block_comment and char == '*' and next_char == '/':
                    in_block_comment = False
                    current_pos += 1
                elif char == '\n':
                    in_line_comment = False
                elif char == '"' or char == "'":
                    if not in_line_comment and not in_block_comment:
                        in_string = not in_string
                
                if not in_line_comment and not in_block_comment and not in_string:
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

            return content[brace_pos:end_brace_pos+1]

        except Exception as e:
            print(f"[ERROR] Failed to extract method body: {e}")
            return None