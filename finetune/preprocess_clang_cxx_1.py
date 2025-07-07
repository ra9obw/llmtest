import os
import json
import re
import itertools
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TokenKind

@dataclass
class CodeElement:
    """Base class for all code elements."""
    type: str
    name: str
    code: str
    documentation: Optional[Dict[str, Any]]
    location: str
    line: int
    context: Optional[str] = None

class CodeExtractor:
    """Improved C++ code structure extractor with better documentation handling."""
    
    DOXYGEN_TAGS = {
        'brief': ('@brief', '\\brief', '\\short'),
        'details': ('@details', '\\details'),
        'param': ('@param', '\\param'),
        'return': ('@return', '@returns', '\\return'),
        'throws': ('@throws', '@exception', '\\throws'),
        'see': ('@see', '\\see'),
        'note': ('@note', '\\note'),
        'warning': ('@warning', '\\warning'),
        'deprecated': ('@deprecated', '\\deprecated'),
        'code': ('@code', '\\code'),
        'endcode': ('@endcode', '\\endcode')
    }
    
    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self.include_dirs = self._find_include_dirs()
        
        self.classes: Dict[str, Dict] = {}
        self.functions: List[Dict] = []
        self.templates: List[Dict] = []
        self.lambdas: List[Dict] = []
        self.macros: List[Dict] = []
        self.error_handlers: List[Dict] = []
        self.comments: List[Dict] = []
        
        self._processed_elements: Set[str] = set()
        self._file_cache: Dict[str, Any] = {}
        self._comment_cache: Dict[Tuple[str, int], Dict] = {}

    def _find_include_dirs(self) -> List[str]:
        """Find all include directories in the repository."""
        include_dirs = [str(self.repo_path)]
        for root, dirs, _ in os.walk(self.repo_path):
            if 'include' in dirs:
                include_dirs.append(str(Path(root) / 'include'))
            if 'inc' in dirs:
                include_dirs.append(str(Path(root) / 'inc'))
        return include_dirs

    def _get_compiler_args(self) -> List[str]:
        """Get compiler arguments for clang parsing."""
        args = [
            '-std=c++17',
            '-x', 'c++',
            '-fparse-all-comments',
            '-D__clang__',
            '-Wno-comment',
            '-Wno-unknown-pragmas',
            '-Wno-macro-redefined'
        ]
        args.extend(f'-I{include_dir}' for include_dir in self.include_dirs)
        return args

    def _get_relative_path(self, absolute_path: str) -> str:
        """Convert absolute path to relative path from repo root."""
        try:
            return str(Path(absolute_path).resolve().relative_to(self.repo_path))
        except ValueError:
            return absolute_path

    def _clean_code(self, code: str) -> str:
        """Clean code while preserving meaningful indentation and comments."""
        if not code:
            return code

        lines = [line.rstrip() for line in code.splitlines()]
        
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Normalize indentation
        lines = [line.replace('\t', '    ') for line in lines]
        
        # Remove excessive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return '\n'.join(cleaned_lines)

    def _get_tokens(self, cursor) -> List[Any]:
        """Get tokens for the given cursor."""
        try:
            return list(cursor.get_tokens())
        except Exception as e:
            print(f"[WARNING] Failed to get tokens for {cursor.spelling}: {e}")
            return []

    def _get_code_snippet(self, cursor) -> Optional[str]:
        """Extract code snippet for the given cursor with preceding comments."""
        if not cursor.location.file:
            return None
            
        try:
            with open(cursor.location.file.name, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            start_line = max(0, cursor.extent.start.line - 1)
            end_line = cursor.extent.end.line
            code = ''.join(lines[start_line:end_line])
            
            # Get preceding comments
            comments = self._get_preceding_comments(cursor)
            if comments:
                code = '\n'.join(comments) + '\n' + code
                
            return self._clean_code(code)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {cursor.location.file}: {e}")
            return None

    def _get_preceding_comments(self, cursor) -> List[str]:
        """Get comments immediately preceding the given cursor."""
        if not cursor.location.file:
            return []
            
        file_path = cursor.location.file.name
        line = cursor.location.line
        comments = []
        
        # Check cached comments first
        for (f, l), comment in self._comment_cache.items():
            if f == file_path and l < line and l >= line - 10:  # Look 10 lines back
                comments.append(comment['text'])
        
        # If not in cache, parse the file
        if not comments:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                
                # Collect comments from previous lines
                current_line = line - 1
                while current_line >= 1:
                    line_content = lines[current_line - 1].strip()
                    if not line_content:
                        current_line -= 1
                        continue
                    
                    if line_content.startswith(('//', '/*', '/**', '/*!', '///', '//!')):
                        comments.insert(0, line_content)
                        current_line -= 1
                    else:
                        break
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")
        
        return comments

    def _parse_doxygen(self, raw_comment: str) -> Dict[str, Any]:
        """Parse Doxygen comment with improved handling."""
        if not raw_comment:
            return {}
        
        # Normalize line endings and remove comment markers
        clean_comment = re.sub(r'^[/\*!]+\s*|\s*[\*/]+$', '', raw_comment, flags=re.MULTILINE)
        clean_comment = clean_comment.strip()
        
        result = {
            'brief': '',
            'details': '',
            'params': {},
            'returns': '',
            'throws': [],
            'see': [],
            'notes': [],
            'warnings': [],
            'deprecated': '',
            'code_blocks': [],
            'raw': raw_comment
        }
        
        # Extract main description before any tags
        first_tag_pos = len(clean_comment)
        for alias in itertools.chain(*self.DOXYGEN_TAGS.values()):
            if (pos := clean_comment.find(alias)) != -1:
                first_tag_pos = min(first_tag_pos, pos)
        
        main_desc = clean_comment[:first_tag_pos].strip()
        if main_desc:
            # Split into brief (first sentence) and details
            sentences = re.split(r'(?<=[.!?])\s+', main_desc)
            if sentences:
                result['brief'] = sentences[0]
                result['details'] = ' '.join(sentences[1:]) if len(sentences) > 1 else ''
        
        # Process tags
        for tag, aliases in self.DOXYGEN_TAGS.items():
            pattern = r"({})\s+(.*?)(?={}|$)".format(
                '|'.join(re.escape(a) for a in aliases),
                '|'.join(re.escape(a) for a in itertools.chain(*self.DOXYGEN_TAGS.values()))
            )
            matches = re.finditer(pattern, clean_comment, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                content = match.group(2).strip()
                if tag == 'param':
                    # Handle @param [in] name description
                    param_match = re.match(r'(\[.*?\])?\s*(\w+)\s+(.*)', content)
                    if param_match:
                        param_name = param_match.group(2)
                        result['params'][param_name] = {
                            'direction': param_match.group(1) or '',
                            'description': param_match.group(3)
                        }
                elif tag in ('throws', 'see', 'note', 'warning'):
                    result[tag + 's' if tag.endswith('s') else tag + 's'].append(content)
                elif tag == 'code':
                    code_content = re.search(r'(.*?)@endcode', content, re.DOTALL)
                    if code_content:
                        result['code_blocks'].append(code_content.group(1).strip())
                else:
                    result[tag] = content
        
        return result

    def _get_documentation(self, cursor) -> Optional[Dict[str, Any]]:
        """Extract documentation for the given cursor."""
        tokens = self._get_tokens(cursor)
        doxygen_comments = []
        
        for token in tokens:
            if token.kind == TokenKind.COMMENT:
                comment_text = token.spelling
                if any(tag in comment_text.lower() for tag in itertools.chain(*self.DOXYGEN_TAGS.values())):
                    doxygen_comments.append(comment_text)
        
        if not doxygen_comments:
            return None
            
        combined = '\n'.join(doxygen_comments)
        return self._parse_doxygen(combined)

    def _get_element_id(self, cursor, element_type: str) -> str:
        """Generate unique ID for an element."""
        location = self._get_relative_path(cursor.location.file.name)
        return f"{element_type}:{location}:{cursor.location.line}:{cursor.spelling}"

    def _is_processed(self, element_id: str) -> bool:
        """Check if element was already processed."""
        return element_id in self._processed_elements

    def _mark_processed(self, element_id: str) -> None:
        """Mark element as processed."""
        self._processed_elements.add(element_id)

    def _get_context(self, cursor) -> Optional[str]:
        """Get the context/namespace of the cursor."""
        context_parts = []
        parent = cursor.semantic_parent
        
        while parent:
            if parent.kind in (CursorKind.NAMESPACE, CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
                if parent.spelling:
                    context_parts.append(parent.spelling)
            parent = parent.semantic_parent
        
        if context_parts:
            return '::'.join(reversed(context_parts))
        return None

    def _process_node(self, cursor) -> None:
        """Process a single AST node."""
        self._process_comments(cursor)
        
        if cursor.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            self._process_class(cursor)
        elif cursor.kind == CursorKind.CLASS_TEMPLATE:
            self._process_template(cursor)
        elif cursor.kind in (CursorKind.CXX_METHOD, CursorKind.FUNCTION_DECL):
            self._process_method_or_function(cursor)
        elif cursor.kind == CursorKind.FUNCTION_TEMPLATE:
            self._process_template(cursor)
        elif cursor.kind == CursorKind.LAMBDA_EXPR:
            self._process_lambda(cursor)
        elif cursor.kind == CursorKind.MACRO_DEFINITION:
            self._process_macro(cursor)
        elif cursor.kind == CursorKind.CXX_TRY_STMT:
            self._process_error_handler(cursor)

    def _process_class(self, cursor) -> None:
        """Process a class/struct declaration."""
        class_name = cursor.spelling
        element_id = self._get_element_id(cursor, "class")
        
        if self._is_processed(element_id) or not class_name:
            return
            
        self._mark_processed(element_id)
        
        class_data = {
            "type": "class",
            "name": class_name,
            "code": self._get_code_snippet(cursor),
            "documentation": self._get_documentation(cursor),
            "methods": [],
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "context": self._get_context(cursor),
            "template_parameters": []
        }
        
        if cursor.kind == CursorKind.CLASS_TEMPLATE:
            class_data["template_parameters"] = self._get_template_parameters(cursor)
        
        self.classes[class_name] = class_data

    def _get_template_parameters(self, cursor) -> List[Dict]:
        """Extract template parameters."""
        params = []
        for child in cursor.get_children():
            if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                params.append({
                    "type": "type",
                    "name": child.spelling,
                    "default": self._get_template_default_value(child)
                })
            elif child.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
                params.append({
                    "type": "non-type",
                    "name": child.spelling,
                    "type_info": child.type.spelling,
                    "default": self._get_template_default_value(child)
                })
        return params

    def _get_template_default_value(self, cursor) -> Optional[str]:
        """Get default value for template parameter."""
        for child in cursor.get_children():
            if child.kind == CursorKind.TYPE_REF:
                return child.type.spelling
        return None

    def _process_method_or_function(self, cursor) -> None:
        """Process a method or free function definition with improved doc handling."""
        code = self._get_code_snippet(cursor)
        if not code or not cursor.spelling:
            return
            
        element_type = "method" if cursor.kind == CursorKind.CXX_METHOD else "function"
        element_id = self._get_element_id(cursor, element_type)
        
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        context = self._get_context(cursor)
        documentation = self._get_documentation(cursor)
        
        # Get preceding comments if no documentation found
        if not documentation:
            preceding_comments = self._get_preceding_comments(cursor)
            if preceding_comments:
                combined_comment = '\n'.join(preceding_comments)
                documentation = self._parse_doxygen(combined_comment)
        
        # Clean up code by removing attached comments
        clean_code = self._clean_code(code)
        if documentation and documentation.get('raw'):
            # Remove documentation from code
            clean_code = clean_code.replace(documentation['raw'], '').strip()
        
        item = {
            "type": element_type,
            "name": cursor.spelling,
            "code": clean_code,
            "documentation": documentation,
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "context": context,
            "parameters": self._parse_parameters_from_signature(clean_code),
            "return_type": self._parse_return_type(clean_code)
        }
        
        parent = cursor.semantic_parent
        if parent and parent.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL):
            class_name = parent.spelling
            if class_name not in self.classes:
                self._process_class(parent)
            
            # Check for duplicate methods (declaration vs implementation)
            existing_methods = [m for m in self.classes[class_name]["methods"] 
                              if m["name"] == item["name"] and m["location"] == item["location"]]
            
            if not existing_methods:
                self.classes[class_name]["methods"].append(item)
        else:
            self.functions.append(item)

    def _process_comments(self, cursor) -> None:
        """Process comments with better classification."""
        tokens = self._get_tokens(cursor)
        
        for token in tokens:
            if token.kind == TokenKind.COMMENT:
                comment_id = f"comment:{self._get_relative_path(token.location.file.name)}:{token.location.line}"
                
                if not self._is_processed(comment_id):
                    self._mark_processed(comment_id)
                    comment_text = token.spelling.strip()
                    
                    # Skip decorative comments
                    if (len(comment_text) > 10 and 
                        not re.match(r'^[/\*-]+\s*$', comment_text) and
                        not re.match(r'^\s*//\s*={3,}\s*$', comment_text)):
                        
                        # Check if this is a Doxygen comment
                        is_doxygen = any(
                            tag in comment_text.lower() 
                            for tag in itertools.chain(*self.DOXYGEN_TAGS.values())
                        )
                        
                        # Cache the comment for potential attachment to code elements
                        file_path = self._get_relative_path(token.location.file.name)
                        self._comment_cache[(file_path, token.location.line)] = {
                            "type": "doxygen" if is_doxygen else "inline",
                            "text": comment_text,
                            "location": file_path,
                            "line": token.location.line,
                            "context": self._get_context(cursor),
                            "is_standalone": self._is_standalone_comment(token, tokens)
                        }

    def _parse_parameters_from_signature(self, signature: str) -> List[Dict]:
        """Parse function parameters from signature."""
        if '(' not in signature or ')' not in signature:
            return []
        
        params_str = signature.split('(')[1].split(')')[0]
        params = []
        
        for param in params_str.split(','):
            param = param.strip()
            if param:
                parts = [p for p in param.split() if p]
                if len(parts) >= 2:
                    param_name = parts[-1]
                    param_type = ' '.join(parts[:-1])
                    default_value = None
                    
                    if '=' in param:
                        default_parts = param.split('=')
                        param_type = default_parts[0].strip().split()[-1]
                        default_value = '='.join(default_parts[1:]).strip()
                    
                    params.append({
                        "name": param_name,
                        "type": param_type,
                        "default_value": default_value
                    })
        
        return params

    def _parse_return_type(self, signature: str) -> str:
        """Parse return type from function signature."""
        if '(' not in signature:
            return 'void'
        
        before_paren = signature.split('(')[0].strip()
        parts = [p for p in before_paren.split() if p]
        
        if not parts:
            return 'void'
        
        # Handle cases like "const std::string&"
        return_type = []
        for p in parts[:-1]:  # Skip the function name
            if p in ('static', 'inline', 'virtual', 'explicit'):
                continue
            return_type.append(p)
        
        return ' '.join(return_type) if return_type else 'void'

    def _process_template(self, cursor) -> None:
        """Process a function or class template."""
        element_type = "class_template" if cursor.kind == CursorKind.CLASS_TEMPLATE else "function_template"
        element_id = self._get_element_id(cursor, element_type)
        
        if self._is_processed(element_id) or not cursor.spelling:
            return
            
        self._mark_processed(element_id)
        
        template_data = {
            "type": element_type,
            "name": cursor.spelling,
            "code": self._get_code_snippet(cursor),
            "documentation": self._get_documentation(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line,
            "context": self._get_context(cursor),
            "template_parameters": self._get_template_parameters(cursor)
        }
        
        if element_type == "class_template":
            self.classes[cursor.spelling] = template_data
        else:
            self.templates.append(template_data)

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
            "line": cursor.location.line,
            "context": self._get_context(cursor)
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
            "line": cursor.location.line,
            "context": self._get_context(cursor)
        })

    def _process_macro(self, cursor) -> None:
        """Process a macro definition."""
        if not cursor.spelling:
            return
            
        element_id = self._get_element_id(cursor, "macro")
        if self._is_processed(element_id):
            return
            
        self._mark_processed(element_id)
        
        code = self._get_code_snippet(cursor)
        args = []
        
        # Parse macro arguments
        if code and '(' in code.split()[0]:
            try:
                args_str = code.split('(')[1].split(')')[0]
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            except:
                pass
        
        self.macros.append({
            "type": "macro",
            "name": cursor.spelling,
            "args": args,
            "code": code,
            "documentation": self._get_documentation(cursor),
            "location": self._get_relative_path(cursor.location.file.name),
            "line": cursor.location.line
        })

    def _is_standalone_comment(self, comment_token, tokens) -> bool:
        """Check if comment is standalone (not attached to code element)."""
        # Check if next token is not a code element
        for token in tokens:
            if token.location.line > comment_token.location.line:
                if token.kind not in (TokenKind.COMMENT, TokenKind.PUNCTUATION):
                    return False
        return True

    def process_file(self, file_path: Path) -> None:
        """Process a single source file with AST caching."""
        file_key = str(file_path.resolve())
        
        if file_key in self._file_cache:
            tu = self._file_cache[file_key]
        else:
            try:
                tu = self.index.parse(
                    str(file_path),
                    args=self._get_compiler_args(),
                    options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                )
                self._file_cache[file_key] = tu
            except Exception as e:
                print(f"[ERROR] Failed to parse {file_path}: {e}")
                return
        
        if not tu:
            return
            
        # Iterative AST traversal
        stack = [tu.cursor]
        processed = set()
        
        while stack:
            cursor = stack.pop()
            
            if cursor.hash in processed:
                continue
            processed.add(cursor.hash)
            
            try:
                self._process_node(cursor)
                stack.extend(reversed(list(cursor.get_children())))
            except Exception as e:
                print(f"[WARNING] Error processing node {cursor.spelling}: {e}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all extracted code structures in enhanced format."""
        results = []
        
        # Classes and class templates
        for cls in self.classes.values():
            cls_data = {
                "type": cls["type"],
                "name": cls["name"],
                "code": cls["code"],
                "documentation": cls["documentation"],
                "location": cls["location"],
                "line": cls["line"],
                "context": cls["context"],
                "methods": [],
                "template_parameters": cls.get("template_parameters", [])
            }
            
            for method in cls.get("methods", []):
                method_data = {
                    "type": method["type"],
                    "name": method["name"],
                    "code": method["code"],
                    "documentation": method["documentation"],
                    "location": method["location"],
                    "line": method["line"],
                    "context": method["context"],
                    "parameters": method.get("parameters", []),
                    "return_type": method.get("return_type", "void")
                }
                cls_data["methods"].append(method_data)
                
            results.append(cls_data)
        
        # Functions and function templates
        for func in self.functions + self.templates:
            func_data = {
                "type": func["type"],
                "name": func["name"],
                "code": func["code"],
                "documentation": func["documentation"],
                "location": func["location"],
                "line": func["line"],
                "context": func["context"],
                "parameters": func.get("parameters", []),
                "return_type": func.get("return_type", "void"),
                "template_parameters": func.get("template_parameters", [])
            }
            results.append(func_data)
        
        # Other elements
        results.extend({
            "type": item["type"],
            **({"name": item["name"]} if "name" in item else {}),
            **({"code": item["code"]} if "code" in item else {}),
            **({"documentation": item["documentation"]} if "documentation" in item else {}),
            "location": item["location"],
            "line": item["line"],
            **({"context": item["context"]} if "context" in item else {}),
            **({"args": item["args"]} if "args" in item else {}),
            **({"text": item["text"]} if "text" in item else {})
        } for item in self.lambdas + self.error_handlers + self.macros + self.comments)
        
        return results

    def save_to_jsonl(self, output_path: str) -> None:
        """Save the extracted data to a JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.get_results():
                if not item.get('code') and not item.get('documentation') and not item.get('text'):
                    continue
                
                json.dump(item, f, ensure_ascii=False, indent=2)
                f.write('\n')

def main() -> None:
    # Configure libclang path
    Config.set_library_file(r"E:\\Programs\\clang-llvm-windows-msvc\\bin\\libclang.dll")
    
    # Path configuration
    REPO_PATH = r"E:\\work\\llm_test\\codebase\\simple"
    OUTPUT_JSONL = r"E:\\work\\llm_test\\dataset_clang.jsonl"
    
    # Initialize extractor
    extractor = CodeExtractor(REPO_PATH)
    
    # Process files
    file_count = 0
    for root, _, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx", ".hh")):
                file_path = Path(root) / file
                print(f"Processing: {file_path}")
                extractor.process_file(file_path)
                file_count += 1
    
    # Print statistics
    print(f"\nProcessing complete:")
    print(f"- Files processed: {file_count}")
    print(f"- Classes found: {len(extractor.classes)}")
    print(f"- Functions found: {len(extractor.functions)}")
    print(f"- Templates found: {len(extractor.templates)}")
    print(f"- Lambdas found: {len(extractor.lambdas)}")
    print(f"- Macros found: {len(extractor.macros)}")
    print(f"- Error handlers found: {len(extractor.error_handlers)}")
    print(f"- Comments found: {len(extractor.comments)}")
    
    # Save results
    extractor.save_to_jsonl(OUTPUT_JSONL)
    print(f"\nResults saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()