import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TypeKind


class CodeExtractor:
    """Extracts code structures (classes, functions, templates, etc.) from C++ source files."""
    
    def __init__(self, repo_path: str, skip_files_func = None) -> None:
        """Initialize the code extractor with repository path."""
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self.include_dirs = self._find_include_dirs()
        self.skip_files_func = skip_files_func
        
        # Data storage
        self.classes: Dict[str, Dict] = {}
        self.class_templates: Dict[str, Dict] = {}
        self.functions: List[Dict] = []
        self.templates: List[Dict] = []
        self.namespaces: List[Dict] = []
        self.lambdas: List[Dict] = []
        self.macros: List[Dict] = []
        self.preprocessor_directives: List[Dict] = []
        self.error_handlers: List[Dict] = []
        self.literals: List[Dict] = []
        self.attributes: List[Dict] = []
        
        # Track all processed elements to avoid duplicates
        self._processed_elements = set()
        
        # Track unprocessed cursor kinds
        self.unprocessed_expected: Dict[str, int] = {}
        self.unprocessed_unexpected: Dict[str, int] = {}
        self.expected_unprocessed = (
                                        CursorKind.DECL_REF_EXPR,
                                        CursorKind.UNEXPOSED_EXPR,
                                        CursorKind.CALL_EXPR,
                                        CursorKind.PARM_DECL,
                                        CursorKind.TYPE_REF,
                                        CursorKind.NAMESPACE_REF,
                                        CursorKind.CXX_ACCESS_SPEC_DECL,
                                        CursorKind.COMPOUND_STMT,
                                        CursorKind.STRING_LITERAL,
                                        CursorKind.TEMPLATE_TYPE_PARAMETER,
                                        CursorKind.DECL_STMT,
                                        CursorKind.VAR_DECL,
                                        CursorKind.TEMPLATE_REF
                                    )

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
        return file_path.startswith('/usr/include') or \
               file_path.startswith('/usr/local/include') or \
               file_path.startswith('/Applications/Xcode.app/') or \
               file_path.startswith('C:\\Program Files (x86)\\Microsoft Visual Studio') or \
               file_path.startswith('C:\\Program Files\\Microsoft Visual Studio') or \
               '\\Windows Kits\\' in file_path or \
               file_path.startswith('<')
    
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

    def _get_code_snippet(self, cursor) -> Optional[str]:
        """Extract code snippet for the given cursor with line and column precision."""
        if not cursor.location.file:
            return None
            
        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            start_line = cursor.extent.start.line - 1  # Convert to 0-based index
            start_col = cursor.extent.start.column - 1
            end_line = cursor.extent.end.line - 1
            end_col = cursor.extent.end.column
            
            # For single-line elements
            if start_line == end_line:
                line = lines[start_line]
                code = line[start_col:end_col]
            else:
                # First line
                code = [lines[start_line][start_col:]]
                
                # Intermediate lines (if any)
                for line_num in range(start_line + 1, end_line):
                    code.append(lines[line_num])
                
                # Last line
                if end_line < len(lines):
                    code.append(lines[end_line][:end_col])
                
                code = ''.join(code)
            
            return self._clean_code(code)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {cursor.location.file}: {e}")
            return None

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

    def _process_unhandled_kind(self, cursor) -> None:
        """Track unhandled cursor kinds."""
        kind_name = str(cursor.kind)
        if cursor.kind in self.expected_unprocessed:
            self.unprocessed_expected[kind_name] = self.unprocessed_expected.get(kind_name, 0) + 1
        else:
            self.unprocessed_unexpected[kind_name] = self.unprocessed_unexpected.get(kind_name, 0) + 1

    def _process_class(self, cursor) -> None:
        """Process a class/struct declaration."""
        class_name = cursor.spelling
        
        element_id = self._get_element_id(cursor, "class")
        
        if self._is_processed(element_id) or class_name in self.classes:
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

    def _process_namespace(self, cursor) -> None:
        """Process a namespace declaration."""
        namespace_name = cursor.spelling or "(anonymous)"
        element_id = self._get_element_id(cursor, "namespace")
        
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.namespaces.append({
            "type": "namespace",
            "name": namespace_name,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _process_class_template(self, cursor) -> None:
        """Process a class template declaration."""
        template_name = cursor.spelling
        if not template_name:  # For anonymous template classes
            template_name = f"anonymous_template_at_{cursor.location.line}"
        
        element_id = self._get_element_id(cursor, "class_template")
        
        if self._is_processed(element_id) or template_name in self.class_templates:
            return
            
        self._mark_processed(element_id)
        
        # Get full template class definition
        code = self._get_code_snippet(cursor)
        
        self.class_templates[template_name] = {
            "type": "class_template",
            "name": template_name,
            "code": code,
            "methods": [],
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "template_parameters": self._get_template_parameters(cursor)
        }

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
                    param["default_value"] = self._get_code_snippet(child)
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
        base_id = self._get_element_id(cursor, "function")
        param_types = [arg.type.spelling for arg in cursor.get_arguments()]
        return f"{base_id}:{':'.join(param_types)}"

    def _process_function(self, cursor) -> None:
        """Process a free function definition with overload support."""
        if not cursor.is_definition():
            return

        element_id = self._generate_overload_id(cursor)
        
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        code = self._get_code_snippet(cursor)
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

        # Check for overloads
        overloads = self._find_function_overloads(cursor)
        if overloads:
            function_info["is_overloaded"] = True
            function_info["overload_count"] = len(overloads) + 1

        self.functions.append(function_info)

    def _find_function_overloads(self, cursor) -> List[Dict]:
        """Find all overloads of the given function with error handling."""
        if not cursor.location.file:
            return []
        
        overloads = []
        seen_signatures = set()
        
        # Get all functions in the same file
        for node in cursor.translation_unit.cursor.walk_preorder():
            try:
                if not hasattr(node, 'kind'):
                    continue
                
                if node.kind != CursorKind.FUNCTION_DECL or node.spelling != cursor.spelling:
                    continue
                    
                if not node.location.file or node.location.file.name != cursor.location.file.name:
                    continue
                if node == cursor:
                    continue
                    
                if not node.is_definition():
                    continue
                    
                try:
                    param_types = []
                    for arg in node.get_arguments():
                        try:
                            param_types.append(arg.type.spelling)
                        except:
                            param_types.append("unknown")
                    param_types = tuple(param_types)
                except:
                    continue
                    
                signature = (node.spelling, param_types)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    try:
                        code = self._get_code_snippet(node)
                        if code:
                            overloads.append({
                                "code": code,
                                "signature": self._get_function_signature(node),
                                "location": self._get_relative_path(node.location.file.name),
                                "line": node.location.line
                            })
                    except:
                        continue
                        
            except ValueError as e:
                if "Unknown template argument kind" in str(e):
                    continue
                raise
            except Exception:
                continue
        
        return overloads

    def _process_class_member(self, cursor) -> None:
        """Process a class/struct/template method with overload support."""
        parent = cursor.semantic_parent
        if not parent:
            return
        
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
        element_id = f"method:{parent_name}::{self._generate_overload_id(cursor)}"
        
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        # Get both declaration and full body for template methods
        code = self._get_code_snippet(cursor)
        full_body = self._get_template_method_body(cursor) if is_template else None
        
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
            
            overloads = self._find_method_overloads(cursor, parent)
            if overloads:
                method_info["is_overloaded"] = True
                method_info["overload_count"] = len(overloads) + 1
            
            if parent_name not in self.class_templates:
                self._process_class_template(parent)
            self.class_templates[parent_name]["methods"].append(method_info)
        else:
            overloads = self._find_method_overloads(cursor, parent)
            if overloads:
                method_info["is_overloaded"] = True
                method_info["overload_count"] = len(overloads) + 1
            
            if parent_name not in self.classes:
                self._process_class(parent)
            self.classes[parent_name]["methods"].append(method_info)

    def _find_method_overloads(self, cursor, parent) -> List[Dict]:
        """Find all overloads of the given method in its parent class with error handling."""
        overloads = []
        seen_signatures = set()
        
        for child in parent.get_children():
            try:
                if not hasattr(child, 'kind'):
                    continue
                
                if child.kind != CursorKind.CXX_METHOD or child.spelling != cursor.spelling:
                    continue
                if child == cursor:
                    continue
                    
                if (parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) and not child.is_definition()):
                    continue
                    
                try:
                    param_types = []
                    for arg in child.get_arguments():
                        try:
                            param_types.append(arg.type.spelling)
                        except:
                            param_types.append("unknown")
                    param_types = tuple(param_types)
                except:
                    continue
                    
                signature = (child.spelling, param_types)
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    try:
                        code = self._get_code_snippet(child)
                        full_body = self._get_template_method_body(child) if parent.kind in (
                            CursorKind.CLASS_TEMPLATE, 
                            CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION
                        ) else None
                        
                        if code or full_body:
                            overloads.append({
                                "code": code,
                                "full_body": full_body if full_body else code,
                                "signature": self._get_function_signature(child),
                                "location": self._get_relative_path(child.location.file.name),
                                "line": child.location.line
                            })
                    except:
                        continue
                        
            except ValueError as e:
                if "Unknown template argument kind" in str(e):
                    continue
                raise
            except Exception:
                continue
        
        return overloads

    def _get_template_parameters(self, cursor) -> List[str]:
        """Extract template parameters from template declaration."""
        params = []
        for child in cursor.get_children():
            if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                params.append(child.spelling)
        return params
    
    def _process_template(self, cursor) -> None:
        """Process a function template."""
        if not (code := self._get_code_snippet(cursor)):
            return
            
        element_id = self._get_element_id(cursor, "template")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        # Get full body for template functions
        full_body = self._get_template_method_body(cursor)
        
        # Check if this is a method template (inside a class/struct/template)
        parent = cursor.semantic_parent
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
                if parent_name not in self.class_templates:
                    self._process_class_template(parent)
                self.class_templates[parent_name]["methods"].append(method_info)
            else:
                if parent_name not in self.classes:
                    self._process_class(parent)
                self.classes[parent_name]["methods"].append(method_info)
        else:
            # Regular function template
            self.templates.append({
                "type": "template",
                "name": cursor.spelling,
                "code": code,
                "full_body": full_body if full_body else code,
                "location": self._get_relative_path(cursor.location.file.name),
                "line": cursor.location.line,
                "template_parameters": self._get_template_parameters(cursor),
                "signature": self._get_function_signature(cursor)
            })

    def _process_lambda(self, cursor) -> None:
        """Process a lambda expression."""
        if not (code := self._get_code_snippet(cursor)):
            return
            
        element_id = self._get_element_id(cursor, "lambda")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.lambdas.append({
            "type": "lambda",
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _process_preprocessor_directive(self, cursor) -> None:
        """Process a preprocessor directive (#ifdef, #pragma, etc.)."""
        if not (code := self._get_code_snippet(cursor)):
            return
            
        element_id = self._get_element_id(cursor, "preprocessor")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.preprocessor_directives.append({
            "type": "preprocessor",
            "directive": cursor.spelling,
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _process_error_handler(self, cursor) -> None:
        """Process a try-catch block."""
        if not (code := self._get_code_snippet(cursor)):
            return
            
        element_id = self._get_element_id(cursor, "error_handler")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.error_handlers.append({
            "type": "error_handler",
            "code": code,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _process_macro(self, cursor) -> None:
        """Process a macro definition."""
        if not cursor.spelling:
            return
            
        element_id = self._get_element_id(cursor, "macro")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.macros.append({
            "type": "macro",
            "name": cursor.spelling,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _process_literal(self, cursor) -> None:
        """Process a user-defined literal."""
        if not cursor.spelling:
            return
            
        element_id = self._get_element_id(cursor, "literal")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.literals.append({
            "type": "literal",
            "name": cursor.spelling,
            "code": self._get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _process_attribute(self, cursor) -> None:
        """Process a C++ attribute ([[...]])."""
        if not cursor.spelling:
            return
            
        element_id = self._get_element_id(cursor, "attribute")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        self.attributes.append({
            "type": "attribute",
            "name": cursor.spelling,
            "code": self._get_code_snippet(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def process_file(self, file_path: Path) -> None:
        """Process a single source file."""
        translation_unit = self.index.parse(
            str(file_path),
            args=self._get_compiler_args()
        )
        
        if not translation_unit:
            return
            
        def visit_node(cursor):
            """Recursively visit AST nodes and process them."""
            if cursor.kind == CursorKind.TRANSLATION_UNIT:
                pass
            else:
                if self._is_system_header(cursor.location.file.name if cursor.location.file else None):
                    return
                
                if self.skip_files_func and self.skip_files_func(self._get_relative_path(cursor.location.file.name)):
                    return
                
                # Class/Struct declarations
                if (cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL) and cursor.is_definition()):
                    self._process_class(cursor)
                # Class templates
                elif cursor.kind in (CursorKind.CLASS_TEMPLATE, CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                    self._process_class_template(cursor)
                # Namespaces
                elif cursor.kind == CursorKind.NAMESPACE:
                    self._process_namespace(cursor)
                # Methods and functions
                elif cursor.kind == CursorKind.CXX_METHOD:
                    self._process_class_member(cursor)
                elif cursor.kind == CursorKind.FUNCTION_DECL:
                    self._process_function(cursor)
                # Templates
                elif cursor.kind == CursorKind.FUNCTION_TEMPLATE:
                    self._process_template(cursor)
                # Lambdas
                elif cursor.kind == CursorKind.LAMBDA_EXPR:
                    self._process_lambda(cursor)
                # Preprocessor directives
                elif cursor.kind == CursorKind.PREPROCESSING_DIRECTIVE:
                    self._process_preprocessor_directive(cursor)
                # Error handlers
                elif cursor.kind == CursorKind.CXX_TRY_STMT:
                    self._process_error_handler(cursor)
                # Macros
                elif cursor.kind == CursorKind.MACRO_DEFINITION:
                    self._process_macro(cursor)
                # User-defined literals
                elif cursor.kind == CursorKind.CONVERSION_FUNCTION:
                    self._process_literal(cursor)
                # Attributes
                elif cursor.kind == CursorKind.ANNOTATE_ATTR:
                    self._process_attribute(cursor)
                else:
                    self._process_unhandled_kind(cursor)
                
            # Process children
            for child in cursor.get_children():
                visit_node(child)
        
        visit_node(translation_unit.cursor)

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all extracted code structures."""
        results = []
        results.extend(cls for cls in self.classes.values() if cls["methods"])
        results.extend(cls_tmpl for cls_tmpl in self.class_templates.values() if cls_tmpl["methods"])
        results.extend(self.functions)
        results.extend(self.templates)
        results.extend(self.namespaces)
        results.extend(self.lambdas)
        results.extend(self.error_handlers)
        results.extend(self.macros)
        results.extend(self.preprocessor_directives)
        results.extend(self.literals)
        results.extend(self.attributes)
        return results


def main() -> None:
    # Укажите путь к libclang.dll
    Config.set_library_file(r"C:\\work\\clang-llvm-20.1.7-windows-msvc\\clang\\bin\\libclang.dll")
    # Настройки путей
    BASE_ROOT = r"C:\\work\\llm_test"
    PROJ_NAME = r"adc4x250"
    # PROJ_NAME = r"simple"
    REPO_PATH = os.path.join(BASE_ROOT, "codebase", PROJ_NAME)
    # REPO_PATH = r"C:\\work\\pavlenko\\llm_test\\codebase\\simple"
    
    # OUTPUT_JSONL = r"C:\\work\\pavlenko\\llm_test\\dataset_clang_test.jsonl"
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
        
    extractor = CodeExtractor(REPO_PATH, should_skip_file)

    file_count = 0
    for root, _, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx")):
                if should_skip_file(file):
                    print(f"Skipping excluded file: {file}")
                    continue

                file_path = Path(root) / file
                print(f"\nProcessing: {file_path}")
                # try:
                if True:
                    extractor.process_file(file_path)
                    file_count += 1
                # except Exception as e:
                #     print(f"[ERROR] Failed to process {file_path}: {e}")
    
    print(f"\nProcessed {file_count} files")
    print(f"Found {len(extractor.classes)} classes")
    print(f"Found {len(extractor.functions)} functions")
    print(f"Found {len(extractor.templates)} templates")
    print(f"Found {len(extractor.class_templates)} class templates")
    print(f"Found {len(extractor.namespaces)} namespaces")
    print(f"Found {len(extractor.lambdas)} lambdas")
    print(f"Found {len(extractor.macros)} macros")
    print(f"Found {len(extractor.preprocessor_directives)} preprocessor directives")
    print(f"Found {len(extractor.literals)} user-defined literals")
    print(f"Found {len(extractor.attributes)} attributes")
    print(f"Found {len(extractor.error_handlers)} error_handlers")
    
    # Print unprocessed kinds statistics
    if extractor.unprocessed_unexpected:
        print("\nUnprocessed cursor kinds:")
        for kind, count in sorted(extractor.unprocessed_unexpected.items(), key=lambda x: x[1], reverse=True):
            print(f"{kind}: {count}")
    else:
        print("\nAll cursor kinds were processed")
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in extractor.get_results():
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()