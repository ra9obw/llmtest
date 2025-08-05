import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TypeKind, StorageClass, Cursor, Diagnostic
from dataclasses import dataclass
import numpy as np

@dataclass
class FragmentConfig:
    target_tokens: int = 512
    chars_per_token: float = 3.5  # Эмпирическое отношение символов к токенам
    min_overlap: int = 50  # Минимальное перекрытие между фрагментами (в символах)
    strategy: str = "ast_priority"  # Может быть "ast_only", "lines", "mixed"
    preserve_comments: bool = True
    include_context: bool = True  # Включать ли контекстную информацию

# class CodeBaseConfig:
#     clang_path = r"C:\\work\\pavlenko\\clang-llvm-windows-msvc-20-1-5\\bin\\libclang.dll"
#     base_root =  r"C:\\work\\pavlenko\\llmtest-git\\codebase\\"
#     proj_name =  r"cppTango-9.3.7"
#     system_includes = [   'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\include',
#                             'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\ucrt',
#                             'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\shared',
#                             'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\um'
#             ]
#     project_includes = [
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\cpp_test_ds',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\cpp_test_ds\\fwd_ds',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\cxxtest\\include\\cxxtest',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\new_tests',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\client',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\client\\helpers',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\server',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\server\\jpeg',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\win32\\resources',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\include\\log4tango',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\include\\log4tango\\threading',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\src',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\tests',
#         'C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\include',
#         'C:\\work\\pavlenko\\omniorb-4.3.0_x64-msvc15_py37\\include',
#         'C:\\work\\pavlenko\\zmq-4.0.5-2_x64-msvc15\\include'
#             ]
#     compile_flags = [
#             '-std=c++14',
#             '-x', 'c++',
#             '-fparse-all-comments',
#             '-D__clang__',
#             '-fno-delayed-template-parsing',
#         ]

class CodeBaseConfig:
    clang_path = r"C:\\work\\clang-llvm-20.1.7-windows-msvc\\clang\\bin\\libclang.dll"
    base_root =  r"C:\\work\\llm_test\\codebase\\"
    proj_name =  r"cppTango-9.3.7"
    system_includes = [   
        'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\14.44.35207\\include',
        'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\ucrt',
        'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\shared',
        'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.22621.0\\um'
            ]
    project_includes = [
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\cpp_test_ds',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\cpp_test_ds\\fwd_ds',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\cxxtest\\include\\cxxtest',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cpp_test_suite\\new_tests',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\client',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\client\\helpers',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\server',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\server\\jpeg',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\cppapi\\win32\\resources',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\include\\log4tango',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\include\\log4tango\\threading',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\src',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\tests',
        'C:\\work\\llm_test\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\log4tango\\include',
        'C:\\work\\llm_test\\codebase\\omniorb-4.3.3_x64-msvc15_py37\\include',
        'C:\\work\\llm_test\\codebase\\zmq-4.0.5-2_x64-msvc15\\include'
            ]
    compile_flags = [
            '-std=c++14',
            '-x', 'c++',
            '-fparse-all-comments',
            '-D__clang__',
            '-fno-delayed-template-parsing',
        ]

class BaseFragmenter:
    def __init__(self, config: FragmentConfig, code_config: CodeBaseConfig):
        self.config = config
        self.code_config = code_config
        self.index_cache = None
        self.index_cached_file = None

    def show_diagnostic(self, translation_unit):
        if translation_unit and translation_unit.diagnostics:
            for diag in translation_unit.diagnostics:
                # Уровень серьёзности (Error, Warning, Note и т. д.)
                severity = diag.severity  # Это число, преобразуем в читаемый формат
                severity_name = {
                    Diagnostic.Error: "Error",
                    Diagnostic.Warning: "Warning",
                    Diagnostic.Note: "Note",
                    Diagnostic.Ignored: "Ignored",
                    Diagnostic.Fatal: "Fatal",
                }.get(diag.severity, f"Unknown ({diag.severity})")
                # Сообщение об ошибке
                message = diag.spelling
                # Позиция в файле (если есть)
                location = diag.location
                file = location.file.name if location.file else "<unknown file>"
                line = location.line
                column = location.column
                # print(f"[{severity_name}] {file}:{line}:{column} - {message}")

    def get_compile_args(self):
        args = self.code_config.compile_flags
        args.extend(arg for include_dir in self.code_config.system_includes 
                   for arg in ['-I', include_dir])
        if self.code_config.project_includes != None:
            args.extend(arg for include_dir in self.code_config.project_includes 
                   for arg in ['-I', include_dir])
        return args

    def parse_code(self, code: str) -> TranslationUnit:
        file_path = self.code_config.base_root + self.code_config.proj_name + "\\" + code["location"]
        if self.index_cached_file and self.index_cached_file == file_path and self.index_cache:
            return self.index_cache
        else:
            index = Index.create()
            args = self.get_compile_args()
            self.index_cache = index.parse(file_path, args=args, options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            self.show_diagnostic(self.index_cache)
            self.index_cached_file = file_path
            return self.index_cache
    
    def get_split_points(self, cursor, code: str) -> List[int]:
        """Нахождение точек разбиения в AST"""
        split_points = []

        def visit(node):
            if node.kind in self._cursor_kinds:
                extent = node.extent
                split_points.append(extent.start.offset)
                split_points.append(extent.end.offset)
            for child in node.get_children():
                visit(child)

        visit(cursor)
        return sorted(set(split_points))

    def calculate_fragments(self, code: str, split_points: List[int]) -> List[Tuple[int, int]]:
        """Вычисление фрагментов с унифицированной нормализацией границ"""
        target_chars = int(self.config.target_tokens * self.config.chars_per_token)
        # print(f"target_chars = {target_chars}\tself.config.chars_per_token = {self.config.chars_per_token}")
        fragments = []
        n = len(code)
        self.full_code_len = n
        start = 0
        split_points = np.array(split_points)

        def find_previous_boundary(pos: int, search_start: int) -> int:
            """Найти ближайшую предыдущую границу (начало строки или ;) начиная с search_start до pos"""
            # Ищем последний перенос строки в диапазоне
            line_start = code.rfind('\n', search_start, pos)
            # Ищем последнюю точку с запятой в диапазоне
            semicolon_pos = code.rfind(';', search_start, pos)
            
            # Выбираем максимальную из найденных границ
            boundary = max(
                line_start + 1 if line_start != -1 else 0,
                semicolon_pos + 1 if semicolon_pos != -1 else 0
            )
            if boundary <= search_start:
                boundary = pos
                # print("No valid boundary found!")
            return boundary

        while True:
            target_end = start + target_chars
            if target_end >= n:
                fragments.append((start, n))
                break
            
            # 1. Приоритет: точки разбиения из AST
            mask = (split_points >= start) & (split_points <= target_end)
            candidates = split_points[mask]
            
            if candidates.any():
                best_split = candidates[-1]
            else:
                # 2. Унифицированный поиск границы
                best_split = find_previous_boundary(target_end, start)
                
                # Если не нашли границ, используем target_end как есть
                if best_split <= start:
                    best_split = target_end

            # Защита от зацикливания
            if best_split <= start + max(self.config.min_overlap, 1):
                best_split = target_end
            
            fragments.append((start, best_split))
            
            # Нормализация новой стартовой позиции
            new_start_raw = max(best_split - self.config.min_overlap, start + 1)
            new_start = find_previous_boundary(new_start_raw, start)
            
            # Защита от зацикливания при нормализации
            if new_start <= start:
                new_start = min(start + 1, n - 1)
            
            start = new_start

        return fragments        
         
    def split(self, code_data: Dict) -> List[Dict]:
        """Основной метод разбиения кода класса"""
        tu = self.parse_code(code_data)        
        # Получаем курсор для всего класса
        class_cursor = None
        for cursor in tu.cursor.get_children():
            if cursor.spelling == code_data['name']:
                class_cursor = cursor
                break
        
        # if not class_cursor:
            # raise ValueError("Class cursor not found in AST")
        code = code_data["code"]
        if class_cursor:
            split_points = self.get_split_points(class_cursor, code)
        else: 
            split_points = []
        # print(f"split_points: {split_points}")
        fragments = self.calculate_fragments(code, split_points)
        
        result = []
        for i, (start, end) in enumerate(fragments):
            fragment_code = code[start:end]
            result.append({
                'element_id': code_data['id'],
                'fragment_id': f"{code_data['id']}_frag{i}",
                'code': fragment_code,
                'start_pos': start,
                'end_pos': end,
                'context': {
                    'element_name': code_data['name'],
                    'file': code_data['location'],
                    'line': code_data['line']
                }
            })
        
        return result

class ClassFragmenter(BaseFragmenter):
    def __init__(self, config: FragmentConfig, codebase_conf: CodeBaseConfig):
        super().__init__(config, codebase_conf)
        self._cursor_kinds = {
            CursorKind.CONSTRUCTOR,
            CursorKind.DESTRUCTOR,
            CursorKind.FUNCTION_TEMPLATE,
            CursorKind.FUNCTION_DECL,
            CursorKind.CXX_METHOD,
            CursorKind.FIELD_DECL,
            CursorKind.DECL_STMT,
            CursorKind.CXX_ACCESS_SPEC_DECL,
            CursorKind.STRUCT_DECL,
            CursorKind.ENUM_DECL,
            CursorKind.ENUM_CONSTANT_DECL,
            CursorKind.TEMPLATE_TYPE_PARAMETER,
            CursorKind.TEMPLATE_TEMPLATE_PARAMETER,
            CursorKind.USING_DECLARATION,
            CursorKind.TYPE_ALIAS_DECL,
            CursorKind.FRIEND_DECL
        }
            
class MethodFragmenter(BaseFragmenter):
    def __init__(self, config: FragmentConfig, codebase_conf: CodeBaseConfig):
        super().__init__(config, codebase_conf)
        self._cursor_kinds = {
            CursorKind.IF_STMT,
            CursorKind.SWITCH_STMT,
            CursorKind.FOR_STMT,
            CursorKind.WHILE_STMT,
            CursorKind.DO_STMT,
            CursorKind.CASE_STMT,
            CursorKind.DEFAULT_STMT,
            CursorKind.DECL_STMT,
            # CursorKind.RETURN_STMT,
            CursorKind.LAMBDA_EXPR,
            CursorKind.CXX_FOR_RANGE_STMT,
            CursorKind.COMPOUND_STMT,
            CursorKind.CXX_TRY_STMT,
            CursorKind.CXX_CATCH_STMT
        }

class ShowCursors(BaseFragmenter):
    def __init__(self, config: FragmentConfig, codebase_conf: CodeBaseConfig):
        super().__init__(config, codebase_conf)
        self._cursor_kinds = { }

    def split(self, code_data: Dict) -> List[Dict]:
        """Основной метод разбиения кода класса"""
        self.code = code_data['code']
        tu = self.parse_code(self.code)
        
        # Получаем курсор для всего класса
        class_cursor = None
        for cursor in tu.cursor.get_children():
            if cursor.spelling == code_data['name']:
                class_cursor = cursor
                break
        
        if not class_cursor:
            raise ValueError("Class cursor not found in AST")
        
        self.show_split_points(class_cursor, self.code)

    def show_split_points(self, cursor, code: str) -> List[int]:
        def visit(node, idx):
            print(f"{idx}: {idx*" "} {node.kind}")
            for child in node.get_children():
                visit(child, idx+1)
        visit(cursor, 0)

class VarFragmenter(BaseFragmenter):
    def __init__(self, config: FragmentConfig, codebase_conf: CodeBaseConfig):
        super().__init__(config, codebase_conf)
        self._cursor_kinds = { }
    
    def get_split_points(self, cursor, code: str) -> List[int]:
        return []
    
    def calculate_fragments(self, code: str, split_points: List[int]) -> List[Tuple[int, int]]:
        """Вычисление фрагментов с унифицированной нормализацией границ"""
        target_chars = int(self.config.target_tokens * self.config.chars_per_token)
        fragments = []
        n = len(code)
        self.full_code_len = n
        start = 0

        def find_previous_boundary(pos: int, search_start: int) -> int:
            """Найти ближайшую предыдущую границу (начало строки или ;) начиная с search_start до pos"""
            # Ищем последний перенос строки в диапазоне
            line_start = code.rfind('\n', search_start, pos)
            # Ищем последнюю точку с запятой в диапазоне
            semicolon_pos = code.rfind(',', search_start, pos)
            
            # Выбираем максимальную из найденных границ
            boundary = max(
                line_start + 1 if line_start != -1 else 0,
                semicolon_pos + 1 if semicolon_pos != -1 else 0
            )
            if boundary <= search_start:
                boundary = pos
                # print("No valid boundary found!")
            return boundary

        while True:
            target_end = start + target_chars
            if target_end >= n:
                fragments.append((start, n))
                break
            best_split = find_previous_boundary(target_end, start)
            if best_split <= start:
                best_split = target_end
            fragments.append((start, best_split))
            start = best_split
        return fragments     


class CodeFragmenter:
    def __init__(self, config: Optional[FragmentConfig] = None, codebase_conf: Optional[CodeBaseConfig] = None):
        self.config = config or FragmentConfig()
        self.codebase_conf = codebase_conf or CodeBaseConfig()
        # CLANG_PATH = r"C:\\work\\clang-llvm-20.1.7-windows-msvc\\clang\\bin\\libclang.dll"
        # CLANG_PATH = r"C:\\work\\pavlenko\\clang-llvm-windows-msvc-20-1-5\\bin\\libclang.dll"
        print(self.codebase_conf.clang_path)
        Config.set_library_file(self.codebase_conf.clang_path)
        self._fragmenters = {
            "class_decl": ClassFragmenter(self.config, self.codebase_conf),
            "struct_decl": ClassFragmenter(self.config, self.codebase_conf),
            "function_decl": MethodFragmenter(self.config, self.codebase_conf),
            "constructor": MethodFragmenter(self.config, self.codebase_conf),
            "destructor": MethodFragmenter(self.config, self.codebase_conf),
            "cxx_method": MethodFragmenter(self.config, self.codebase_conf),
            "class_template": ClassFragmenter(self.config, self.codebase_conf),
            "function_template": MethodFragmenter(self.config, self.codebase_conf),
            "var_decl": VarFragmenter(self.config, self.codebase_conf)
        }
        
    def split_code(self, code_data: Dict) -> List[Dict]:
        """Универсальный метод для разбиения любого типа кода"""
        entity_type = code_data.get("type", "")
        fragmenter = self._fragmenters.get(entity_type)
        if not fragmenter:
            raise ValueError(f"Unsupported code type: {entity_type}")
            
        return fragmenter.split(code_data)
    
    def visualize_fragments(self, fragments: List[Dict], code_data: Dict):
        _code = code_data['code']
        for frag in fragments:
            print(f"\nFragment {frag['fragment_id']} ({frag['end_pos']-frag['start_pos']} chars):")
            print("="*50)
            print(_code[frag['start_pos']:frag['end_pos']])
            print("="*50)

    def evaluate_fragmentation(self, fragments: List[Dict], code_data: Dict):
        _code = code_data['code']
        sizes = [f['end_pos'] - f['start_pos'] for f in fragments]
        avg_size = sum(sizes) / len(sizes)
        target = self.config.target_tokens * self.config.chars_per_token
        print(f"Fragmentation metrics:")
        print(f"- Total fragments: {len(fragments)}")
        if len(sizes) > 1:
            print(f"- Avg size: {avg_size:.1f} chars (target: {target})")
            print(f"- Size stddev: {np.std(sizes[:-1]):.1f}\tmin: {np.min(sizes[:-1])}\tmax: {np.max(sizes[:-1])}")
            print(f"- Coverage: {sum(sizes)/len(_code)*100:.1f}%")


class SignatureExtractor:
    def __init__(self):
        pass
    
    def _find_closing_parenthesis(self, lines, start_line, start_col):
        """
        Находит позицию закрывающей скобки ')', начиная с указанной позиции.
        Возвращает кортеж (line, column) или (None, None), если скобка не найдена.
        Учитывает вложенные скобки и игнорирует скобки в строковых литералах и комментариях.
        """
        line_num = start_line
        col_num = start_col
        paren_level = 0  # Уровень вложенности скобок
        in_string = False  # Флаг нахождения внутри строкового литерала
        in_comment = False  # Флаг нахождения внутри комментария
        
        while line_num < len(lines):
            line = lines[line_num]
            
            # Если это не первая строка, начинаем с начала строки
            search_start = col_num if line_num == start_line else 0
            
            for i in range(search_start, len(line)):
                char = line[i]
                
                # Обработка строковых литералов
                if char == '"' and not in_comment:
                    in_string = not in_string
                    continue
                    
                # Обработка комментариев
                if not in_string:
                    if char == '/' and i + 1 < len(line) and line[i+1] == '*':
                        in_comment = True
                    elif char == '*' and i + 1 < len(line) and line[i+1] == '/':
                        in_comment = False
                        continue
                    elif char == '/' and i + 1 < len(line) and line[i+1] == '/':
                        break  # Пропускаем оставшуюся часть строки
                        
                if in_string or in_comment:
                    continue
                    
                # Подсчет скобок
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                    if paren_level <= 0:
                        return (line_num, i)
                    
            # Переход к следующей строке
            line_num += 1
            col_num = 0
        
        print(") not found")
        return (max(0, len(lines)-1), max(0,len(lines[-1])-1))  # Скобка не найдена

    def _extract_code_fragment(self, lines, start_line, start_col, stop_line, stop_col):
        """
        Извлекает фрагмент кода из lines между указанными позициями
        Возвращает строку с извлеченным фрагментом
        """
        extracted = []
        
        # Обрабатываем однострочный случай
        if start_line == stop_line:
            line = lines[start_line]
            return line[start_col:stop_col].strip()
        
        # Многострочный случай
        # Первая строка (от start_col до конца строки)
        first_line = lines[start_line][start_col:]
        extracted.append(first_line)
        
        # Промежуточные строки (полностью)
        for line_num in range(start_line + 1, stop_line):
            extracted.append(lines[line_num])
        
        # Последняя строка (от начала до stop_col)
        last_line = lines[stop_line][:stop_col]
        extracted.append(last_line)
        
        # Объединяем и убираем лишние пробелы
        return ' '.join(line.strip() for line in extracted)

    def _get_function_signature(self, cursor) -> Dict[str, Any]:

        lines = self.code.split('\n')

        func_start_line = cursor.extent.start.line - 1  # Переводим в 0-based индекс
        func_start_col = cursor.extent.start.column - 1

        end_line, end_col = self._find_closing_parenthesis(lines, func_start_line, func_start_col)

        children = list(cursor.get_children())
        param_cursors = [c for c in children if c.kind == CursorKind.PARM_DECL]

        if len(param_cursors) == 0:
            func_stop_line = cursor.extent.end.line - 1  # Переводим в 0-based индекс
            func_stop_col = cursor.extent.end.column - 1
        else:
            func_stop_line = param_cursors[0].extent.start.line - 1  # Переводим в 0-based индекс
            func_stop_col = param_cursors[0].extent.start.column - 1
        
        name = cursor.spelling
        _str = self._extract_code_fragment(lines, func_start_line, func_start_col, func_stop_line, func_stop_col)
        frags = _str.split(name, maxsplit=1)
        signature = {
            "name": cursor.spelling,
            "return_type": frags[0].strip(),
            "parameters": [],
            "qualifiers": {
                "is_const": cursor.is_const_method(),
                "is_static": cursor.is_static_method(),
                "is_virtual": cursor.is_virtual_method(),
                "is_pure_virtual": cursor.is_pure_virtual_method(),
                "is_noexcept": "noexcept" in cursor.result_type.spelling,
            }
        }

        for idx, param in enumerate(param_cursors):
            func_start_line = param.extent.start.line - 1
            func_start_col = param.extent.start.column - 1
            if idx == len(param_cursors)-1:
                func_stop_line,func_stop_col = end_line, end_col
            else:
                func_stop_line = param_cursors[idx+1].extent.start.line - 1  # Переводим в 0-based индекс
                func_stop_col = param_cursors[idx+1].extent.start.column - 1
            _str = self._extract_code_fragment(lines, func_start_line, func_start_col, func_stop_line, func_stop_col)
            print(
                func_start_line,
                func_start_col,
                func_stop_line,
                func_stop_col
            )
            print(_str)
            sp = _str.split("=")
            if len(sp) == 2:
                default_value = sp[1].strip()
            else:
                default_value = None
            param_name = param.spelling
            param_type = (sp[0].split(param_name))[0].strip()
        
            
            signature["parameters"].append({
                "name": param_name,
                "type": param_type,
                "default_value": default_value
            })

        return signature
    
    def parse_code(self, code: str) -> TranslationUnit:
        index = Index.create()
        return index.parse('tmp.cpp', args=['-std=c++14'], unsaved_files=[('tmp.cpp', code)])
    
    def show_diagnostic(self, translation_unit):
        if translation_unit.diagnostics:
            for diag in translation_unit.diagnostics:
                # Уровень серьёзности (Error, Warning, Note и т. д.)
                severity = diag.severity  # Это число, преобразуем в читаемый формат
                severity_name = {
                    Diagnostic.Error: "Error",
                    Diagnostic.Warning: "Warning",
                    Diagnostic.Note: "Note",
                    Diagnostic.Ignored: "Ignored",
                    Diagnostic.Fatal: "Fatal",
                }.get(diag.severity, f"Unknown ({diag.severity})")

                # Сообщение об ошибке
                message = diag.spelling

                # Позиция в файле (если есть)
                location = diag.location
                file = location.file.name if location.file else "<unknown file>"
                line = location.line
                column = location.column

                print(f"[{severity_name}] {file}:{line}:{column} - {message}")

    def extract(self, code_data: Dict):
        self.code = code_data['code']
        tu = self.parse_code(self.code)
        self.show_diagnostic(tu)
        
        # Получаем курсор для всего класса
        class_cursor = None
        for cursor in tu.cursor.get_children():
            print(cursor.spelling)
            if cursor.spelling == code_data['name']:
                class_cursor = cursor
                break
        
        if not class_cursor:
            raise ValueError("Class cursor not found in AST")
        
        def visit(node, idx):
            print(f"{idx}: {idx*" "} {node.kind}")
            if node.kind in [CursorKind.FUNCTION_DECL]:
                return self._get_function_signature(node)
            for child in node.get_children():
                visit(child, idx+1)
            raise ValueError(f"No {CursorKind.FUNCTION_DECL} found!")
        return visit(cursor, 0)

import re

def clean_cpp_code(code):
    # Удаляем комментарии
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # /* ... */
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)  # // ...
    code = re.sub(r'///.*?$', '', code, flags=re.MULTILINE)  # /// ...
    code = re.sub(r'//!.*?$', '', code, flags=re.MULTILINE)  # //! ...

    # Нормализация пробелов:
    # 1. Заменяем все последовательности пробелов/табов на один пробел
    code = re.sub(r'[ \t]+', ' ', code)
    # 2. Убираем пробелы перед <, >, ::, *, &, (, ), {, }, [, ], ;, ,, ::
    code = re.sub(r' ([<>{}\[\]();,&*:]|::)', r'\1', code)
    # 3. Убираем пробелы после <, >, ::, *, &, (, {, [ 
    code = re.sub(r'([<>{}\[(::*&]) ', r'\1', code)
    # 4. Убираем пробелы в начале/конце строки
    code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())

    return code

# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_cppTango-9.3.7.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_simple.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_adc4x250.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_template_exampl.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_overload_example.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\pavlenko\\llmtest-git\\dataset_clang_test_examples.jsonl")
INPUT_JSONL = Path(f"C:\\work\\pavlenko\\llmtest-git\\dataset_clang_cppTango-9.3.7.jsonl")


def get_code_element(_type = "classes", _name = "DeviceImpl", _file = INPUT_JSONL):
    with open(_file, "r", encoding="utf-8") as in_f:

        for line in in_f:
            entry = json.loads(line)
            # print(len(entry), entry['type'])
            if entry['type'] == _type and entry["name"] == _name and entry["is_defined"] == True:
                
                _desc = f"{entry['type']}:\t{entry["name"]} with type {entry["type"]}"
                # _doc = entry.get("docstring", "")
                _code = entry.get("code", "")
                # _sgntr = entry.get("signature", "")
                # _body =  entry.get("full_body", "")
                # _is_defined = entry.get("is_defined", "None")
                _comment = "".join([el["text"] for el in entry["comments"]])
                # _docstring = "".join([el["text"] for el in entry["docstrings"]])

                print(f"{_desc}")
                print("comment is\n", _comment)
                # print("docstring is", _docstring)
                # print(_doc)
                # print(_sgntr)
                print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}")
                # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, {[len(x) for x in _code.split("\n")]}")
                # print(f"body: {_code}")
                return entry
    print("Nothing Found!")
    return ""

if __name__ == "__main__":
    CLANG_PATH = r"C:\\work\\pavlenko\\clang-llvm-windows-msvc-20-1-5\\bin\\libclang.dll"
    Config.set_library_file(CLANG_PATH)
    # code_data = {}
    # code_data["name"] = "operator<<"
    # code_data["code"] = "inline void operator<<(DevVarCharArray &lval,const vector<unsigned char> &rval = {\"0\"});"
    # code_data["name"] = "show"
    # code_data["code"] = "vector<unsigned char> show (int a, vector<unsigned char>& b = {0, 1}, float c);"
    # code_data["code"] = "const std::vector<int>& \nprocess_data(\n    const std::string& input,\n    int flags = DEFAULT_FLAGS,\n    bool verbose = false\n) noexcept;\n"
    # code_data["name"] = "process_data"
    # code_data["code"] = "vector<unsigned char> show();"
    # code_data["name"] = "deep_copy"
    # code_data["code"] = "void DeviceAttribute::deep_copy(const DeviceAttribute & source)\n{\n    w_dim_x = source.w_dim_x;\n    w_dim_y = source.w_dim_y;\n\n}"

    # sh = SignatureExtractor()
    # print(sh.extract(code_data))
    
        # Загрузка токенизатора
    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    # _code_element = get_code_element(_type="structures", _name = "TimeStampComponent")
    # _code_element = get_code_element(_type="class_templates", _name = "TimedAttrData")
    _code_element = get_code_element(_type="cxx_method", _name = "set_max_warning")
    # _code_element = get_code_element(_type="function_template", _name = "operator>>")
    # _code_element = get_code_element(_type="class_template", _name = "DataElement")
    # _code_element = get_code_element(_type="var_decl", _name = "val_ac_luminance")
    
    # _code_element = get_code_element()
    # print(type(_code_element))
    # _code = _code_element["code"]

    config = FragmentConfig(
        target_tokens=512,
        chars_per_token=3.333,  # Для C++ обычно 3.0-3.5
        min_overlap=30
    )
    code_config = CodeBaseConfig()
    fragmenter = CodeFragmenter(config, code_config)
    fragments = fragmenter.split_code(_code_element)
    fragmenter.evaluate_fragmentation(fragments, _code_element)

    # input_tokens_no_trunc = tokenizer(_code, truncation=False)
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}")
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(_code)/len(input_tokens_no_trunc[0])}")

    # _code = clean_cpp_code(_code)
    # _code_element["code"] = _code
    # fragments = fragmenter.split_code(_code_element)
    # fragmenter.evaluate_fragmentation(fragments, _code_element)
    # # input_tokens_no_trunc = tokenizer(_code, truncation=False)
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}")
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(_code)/len(input_tokens_no_trunc[0])}")

    # fragmenter.visualize_fragments(fragments, _code_element)


    # from transformers import (
    # AutoTokenizer
    # )
    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    # for fr in fragments:
    #     input_tokens_no_trunc = tokenizer(fr['code'], truncation=False)
    #     print(f"tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(fr['code'])/len(input_tokens_no_trunc[0])}")