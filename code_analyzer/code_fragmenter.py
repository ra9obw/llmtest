import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from clang.cindex import Index, CursorKind, Config, TranslationUnit, TypeKind, StorageClass
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

class BaseFragmenter:
    def __init__(self, config: FragmentConfig):
        self.config = config
        
    def parse_code(self, code: str) -> TranslationUnit:
        index = Index.create()
        return index.parse('tmp.cpp', args=['-std=c++17'], unsaved_files=[('tmp.cpp', code)])
    
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
                print("No valid boundary found!")
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
        code = code_data['code']
        tu = self.parse_code(code)
        
        # Получаем курсор для всего класса
        class_cursor = None
        for cursor in tu.cursor.get_children():
            if cursor.spelling == code_data['name']:
                class_cursor = cursor
                break
        
        if not class_cursor:
            raise ValueError("Class cursor not found in AST")
        
        split_points = self.get_split_points(class_cursor, code)
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
    def __init__(self, config: FragmentConfig):
        super().__init__(config)
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
    def __init__(self, config: FragmentConfig):
        super().__init__(config)
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
    def __init__(self, config: FragmentConfig):
        super().__init__(config)
        self._cursor_kinds = { }

    def split(self, code_data: Dict) -> List[Dict]:
        """Основной метод разбиения кода класса"""
        code = code_data['code']
        tu = self.parse_code(code)
        
        # Получаем курсор для всего класса
        class_cursor = None
        for cursor in tu.cursor.get_children():
            if cursor.spelling == code_data['name']:
                class_cursor = cursor
                break
        
        if not class_cursor:
            raise ValueError("Class cursor not found in AST")
        
        self.show_split_points(class_cursor, code)

    def show_split_points(self, cursor, code: str) -> List[int]:
        def visit(node, idx):
            print(f"{idx}: {idx*" "} {node.kind}")
            for child in node.get_children():
                visit(child, idx+1)
        visit(cursor, 0)

class VarFragmenter(BaseFragmenter):
    def __init__(self, config: FragmentConfig):
        super().__init__(config)
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
                print("No valid boundary found!")
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
    def __init__(self, config: Optional[FragmentConfig] = None):
        self.config = config or FragmentConfig()
        self._fragmenters = {
            "class_decl": ClassFragmenter(self.config),
            "struct_decl": ClassFragmenter(self.config),
            "function_decl": MethodFragmenter(self.config),
            "constructor": MethodFragmenter(self.config),
            "destructor": MethodFragmenter(self.config),
            "cxx_method": MethodFragmenter(self.config),
            "class_template": ClassFragmenter(self.config),
            "function_template": MethodFragmenter(self.config),
            "var_decl": VarFragmenter(self.config)
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
    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:

        for line in in_f:
            entry = json.loads(line)
            # print(len(entry), entry['type'])
            if entry["data"]['type'] == _type and entry["data"]["name"] == _name:
                
                _desc = f"{entry['type']}:\t{entry["data"]["name"]} with type {entry["data"]["type"]}"
                # _doc = entry.get("docstring", "")
                # _code = entry["data"].get("code", "")
                # _sgntr = entry.get("signature", "")
                # _body =  entry["data"].get("full_body", "")
                # _is_defined = entry["data"].get("is_defined", "None")
                # _comment = "".join([el["text"] for el in entry["data"]["comments"]])
                # _docstring = "".join([el["text"] for el in entry["data"]["docstrings"]])

                print(f"{_desc}")
                # print("comment is\n", _comment)
                # print("docstring is", _docstring)
                # print(_doc)
                # print(_sgntr)
                # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, {[len(x) for x in _code.split("\n")]}")
                # print(f"body: {_body}")
                return entry["data"]
    return ""

if __name__ == "__main__":
    from settings import settings
    Config.set_library_file(settings["CLANG_PATH"])
        # Загрузка токенизатора
    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    # _code_element = get_code_element(_type="structures", _name = "TimeStampComponent")
    # _code_element = get_code_element(_type="class_templates", _name = "TimedAttrData")
    # _code_element = get_code_element(_type="methods", _name = "DeviceAttribute")
    # _code_element = get_code_element(_type="function_template", _name = "operator>>")
    # _code_element = get_code_element(_type="class_template", _name = "DataElement")
    _code_element = get_code_element(_type="var_decl", _name = "val_ac_luminance")
    
    # _code_element = get_code_element()
    print(type(_code_element))
    _code = _code_element["code"]

    config = FragmentConfig(
        target_tokens=512,
        chars_per_token=1.1,  # Для C++ обычно 3.0-3.5
        min_overlap=30
    )
    fragmenter = CodeFragmenter(config)
    fragments = fragmenter.split_code(_code_element)
    fragmenter.evaluate_fragmentation(fragments, _code_element)

    # input_tokens_no_trunc = tokenizer(_code, truncation=False)
    print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}")
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(_code)/len(input_tokens_no_trunc[0])}")

    # _code = clean_cpp_code(_code)
    # _code_element["code"] = _code
    # fragments = fragmenter.split_code(_code_element)
    # fragmenter.evaluate_fragmentation(fragments, _code_element)
    # # input_tokens_no_trunc = tokenizer(_code, truncation=False)
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}")
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(_code)/len(input_tokens_no_trunc[0])}")

    fragmenter.visualize_fragments(fragments, _code_element)


    from transformers import (
    AutoTokenizer
    )
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    for fr in fragments:
        input_tokens_no_trunc = tokenizer(fr['code'], truncation=False)
        print(f"tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(fr['code'])/len(input_tokens_no_trunc[0])}")