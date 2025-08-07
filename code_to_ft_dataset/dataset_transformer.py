import re
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer
    )
from dataclasses import dataclass
from code_fragmenter import CodeFragmenter
from cxx_signature_generator import generate_signature
from cxx_instruction_generator import (
    GenerationMode,
    generate_instruction
)
from clang.cindex import Config
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class DsTokenCounterConfig:
    model_name: str = "Qwen/Qwen3-0.6B"


class DsTokenCounter:
    """Counts tokens in text using a pretrained tokenizer."""
    
    def __init__(self, config: DsTokenCounterConfig = DsTokenCounterConfig()):
        self._config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name, 
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer: {str(e)}")

    def get_token_count(self, in_str: str) -> int:
        """Get token count for a single string."""
        _input_tokens_no_trunc = self.tokenizer(in_str, truncation=False)
        return len(_input_tokens_no_trunc["input_ids"])
    
    def get_batch_token_count(self, in_str: List[str]) -> List[int]:
        """Get token counts for a batch of strings."""
        _input_tokens_no_trunc = self.tokenizer(in_str, truncation=False)
        return [len(_inp) for _inp in _input_tokens_no_trunc["input_ids"]]


class DsTransformerBase:
    """Base class for code transformers."""
    
    def __init__(
        self, 
        max_tokens: int = 512, 
        tkn: Optional[DsTokenCounter] = None, 
        fragmenter: Optional[CodeFragmenter] = None,
        deration: float = 0.75
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._tkn = tkn
        self.fragmenter = fragmenter
        self.code_writer = []
        self.documenter = []
        self.too_long_inp_counter = 0
        self.skipped_counter = 0
        self.input_part_order = ['instruction', 'location', 'context_before', 'comments', 'docstring']
        self.removal_priority = ['comments', 'context_before', 'location']

    def clean_docstring(self, docstring):
        # Удаляем строки, состоящие только из * или - (рамки)
        cleaned = re.sub(r'^[*+-]+$', '', docstring, flags=re.MULTILINE)
        # Удаляем начальные /* и */
        cleaned = re.sub(r'^/\*\*|\*/', '', cleaned, flags=re.MULTILINE)
        # Удаляем все * в начале строки (с возможными пробелами перед ними)
        cleaned = re.sub(r'^\s*\* ?', '', cleaned, flags=re.MULTILINE)
        # Удаляем строки, состоящие только из - (разделители)
        cleaned = re.sub(r'^-+$', '', cleaned, flags=re.MULTILINE)
        # Удаляем множественные пробелы (оставляем один)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        # Удаляем пробелы перед и после переносов строк
        cleaned = re.sub(r' *\n *', '\n', cleaned)
        # Заменяем множественные переносы строк на один
        cleaned = re.sub(r'\n{2,}', '\n', cleaned)
        cleaned = re.sub(r'^[*+-]+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n{2,}', '\n', cleaned)
        # Удаляем начальные и конечные переносы строк и пробелы
        cleaned = cleaned.strip()
        return cleaned

    def clean_comment(self, comment):
        lines = comment.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:  # Пропускаем пустые строки
                continue
                
            # Пропускаем разделители и пустые комментарии
            if re.fullmatch(r'//[-+=]+\s*', stripped_line):
                continue
            if re.fullmatch(r'//[-+]*\s*', stripped_line):
                continue
                
            # Удаляем "//", "//-", "//+" в начале
            useful_part = re.sub(r'^//[-+]*\s*', '', stripped_line)
            # Удаляем @{ и @}
            useful_part = re.sub(r'@[{}]', '', useful_part).strip()  # Дополнительно strip() на случай пробелов
            
            # Пропускаем строку, если после всех замен она пустая или состоит из пробелов
            if not useful_part:
                continue
                
            cleaned_lines.append(useful_part)
        
        # Объединяем строки и удаляем лишние пробелы
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def clean_cpp_code(self, code):
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # /* ... */
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)  # // ...
        code = re.sub(r'///.*?$', '', code, flags=re.MULTILINE)  # /// ...
        code = re.sub(r'//!.*?$', '', code, flags=re.MULTILINE)  # //! ...
        code = re.sub(r'[ \t]+', ' ', code)
        code = re.sub(r' ([<>{}\[\]();,&*:]|::)', r'\1', code)
        code = re.sub(r'([<>{}\[(::*&]) ', r'\1', code)
        code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())
        return code
        
    def _do_fullcode_instruction(self, element: Dict) -> str:
        return generate_instruction(element["definition"], GenerationMode.FULL)
    
    def _do_fragcode_header(self, element: Dict) -> str:
        return generate_instruction(element["definition"], GenerationMode.FIRST_FRAGMENT)

    def _do_fragcode_body(self, idx: str, element: Dict) -> str:
        return generate_instruction(element["definition"], GenerationMode.NEXT_FRAGMENT, idx)

    def _do_fragcode_tail(self, element: Dict) -> str:
        return generate_instruction(element["definition"], GenerationMode.LAST_FRAGMENT)    

    def _do_docsting_instruction(self, element: Dict) -> str:
        return generate_instruction(element["definition"], GenerationMode.DOCSTRING)
    
    def generate_full_name(self, element):
        pass
    
    def _get_docstring(self, element):
        if element["definition"]["docstrings"]:
            return self.clean_docstring('\n'.join([doc['text'] for doc in element["definition"]['docstrings']]))
        return None

    def _add_doc_and_comment(self, element):
        parts = {}
        _doc = self._get_docstring(element)
        if _doc: 
            parts["docstring"] = f" using docstring: \"{_doc}\""
        if element["definition"]["comments"]:
            _comm = self.clean_comment(''.join(el['text'] for el in element["definition"]['comments']))
            if _comm: 
                parts["comments"] = f" using comments: \"{_comm}\""
        return parts

    def _build_final_input(self, parts: Dict[str, str]) -> str:
        return "".join(parts.get(part_name, "") for part_name in self.input_part_order if part_name in parts)

    def _build_truncated_input(self, parts: Dict[str, str]) -> str:
        """Собирает итоговую строку из компонентов с учетом ограничения на токены.
        
        Args:
            parts: Словарь с компонентами строки (ключ - тип компонента, значение - текст)
            max_tokens: Максимальное допустимое количество токенов
        
        Returns:
            Собранная строка, уложившаяся в лимит токенов
        """
        if self.max_tokens is None:
            return "".join(parts.values())
        
        # Рассчитываем длину каждого компонента в токенах
        component_lengths = {k: self._tkn.get_token_count(v) for k, v in parts.items()}
        
        total_tokens = sum(component_lengths.values())
        
        # Если укладываемся в лимит, возвращаем как есть
        if total_tokens <= self.max_tokens:
            return "".join(parts.values())
        # Создаем копию, чтобы не изменять исходный словарь
        remaining_parts = parts.copy()
        # Удаляем компоненты по приоритету, пока не уложимся в лимит
        for component in self.removal_priority:
            if component in remaining_parts.keys() and total_tokens > self.max_tokens:
                del remaining_parts[component]
                total_tokens -= component_lengths[component]
        
        # Если после удаления всех необязательных компонентов все равно превышаем лимит,
        # обрезаем оставшуюся строку
        if total_tokens > self.max_tokens:
            self.too_long_inp_counter += 1
            print(f"still too long {component_lengths} {parts["location"]}")
        
        return "".join(remaining_parts.values())

    def _header(self, element: Dict, is_full: bool = True) -> Dict:
        """Generate header part of the transformed class code."""
        _instruction = (self._do_fullcode_instruction(element) if is_full 
                       else self._do_fragcode_header(element))
        parts = {"instruction": _instruction,
        "location": f" located at file: {element["definition"]['location']}, line:{element["definition"]['line']}"
        }
        
        if element["definition"]["context_before"]:
            parts["context_before"] = f"\nContext before is: \"{'\n'.join(element["definition"]['context_before'])}\""

        parts.update(self._add_doc_and_comment(element))

        return self._build_truncated_input(parts)
    
    def _body(self, element: Dict, frag: int, is_end: bool = True) -> Dict:
        """Generate body part of the transformed class code."""
        _instruction = (self._do_fragcode_tail(element) if is_end 
                       else self._do_fragcode_body(str(frag), element))
        parts = {"instruction": _instruction,
            "location": f" located at file {element["definition"]['location']}"
        }
        
        parts.update(self._add_doc_and_comment(element))
        return self._build_truncated_input(parts)
    
    def _docstring_generate(self, element: Dict):
        _doc = self._get_docstring(element)
        if _doc:
            _inst = self._do_docsting_instruction(element)
            _code = element["definition"]["code"]
            _inst += f" with code: {_code}"
            self.documenter.append({"output": _doc, "input": _inst})

    def _process_code_fragments(self, element: Dict):
        
        if self._tkn and self.fragmenter:
            _tkn_count = self._tkn.get_token_count(element["definition"]["code"])
            
        if (not self._tkn or not self.fragmenter) or _tkn_count <= self.max_tokens:
            self.code_writer.append({"input": self._header(element), "output": element["definition"]["code"]})
        else:
            self.fragmenter.config.chars_per_token = 0.8 * len(element["definition"]["code"]) / _tkn_count 
            frags = self.fragmenter.split_code(element["definition"])
            
            for _idx, frag in enumerate(frags):
                if _idx == 0:
                    self.code_writer.append( {"input": self._header(element, is_full=False),
                                             "output": frag["code"]})
                else:
                    self.code_writer.append( {"input": self._body(element, _idx, is_end=(_idx == len(frags)-1)), 
                                             "output": frag["code"]})

    def transform(self, element: Dict) -> Dict:
        """Transform the input element."""
        raise NotImplementedError


class FunctionsTransformer(DsTransformerBase):
    """Generic code transformer that can handle different element types."""
    
    def __init__(self, max_tokens: int = 512, 
                 tkn: Optional[DsTokenCounter] = None, 
                 fragmenter: Optional[CodeFragmenter] = None,
                 deration: float = 0.75):
        super().__init__(max_tokens, tkn, fragmenter, deration)
        self.input_part_order = ['instruction', 'signature', 'location', 'context_before', 'comments', 'docstring']

    def _get_parameters(self, signature):
        parameters = []
        for param in signature["parameters"]:
            param_str = f"{param['type']}"
            # if param["name"]:
            #     param_str += f"_{param['name']}"
            # if param["default_value"] is not None:
            #     param_str += f"[{param['default_value']}]"
            parameters.append(param_str)
        return parameters

    def generate_full_name(self, element):
        _name = element["name"]
        if element["parent_type"] != "translation_unit":
            _name = element["parent_type"] + "_" + element["parent_name"] + "_" + _name
        signature = element["signature"]
        if signature["qualifiers"]["is_const"]:
            _name += "_const_"
        # _name += f"_ret_{signature["return_type"]}"
        parameters = self._get_parameters(signature)
        if len(parameters) != 0:
            _name += '_'
            _name += '_'.join(parameters)
        return _name
    
    def _get_docstring(self, element):
        _doc = None
        if element["definition"]["docstrings"]:
            _doc = self.clean_docstring('\n'.join([doc['text'] for doc in element["definition"]['docstrings']]))
        else:
            if element["declaration"] and element["declaration"]["docstrings"]:
                _doc = self.clean_docstring('\n'.join([doc['text'] for doc in element["declaration"]['docstrings']]))
        return _doc

    def _add_doc_and_comment(self, element):
        parts = {}
        _doc = self._get_docstring(element)
        if _doc: 
            parts["docstring"] = f" using docstring: \"{_doc}\""
        if element["definition"]["comments"]:
            _comm = self.clean_comment(''.join(el['text'] for el in element["definition"]['comments']))
            if _comm: 
                parts["comments"] = f" using comments: \"{_comm}\""
        else:
            if element["declaration"] and element["declaration"]["comments"]:
                _comm = self.clean_comment(''.join(el['text'] for el in element["declaration"]['comments']))
                if _comm: 
                    parts["comments"] = f" using comments: \"{_comm}\""
        return parts

    def _header(self, element: Dict, is_full: bool = True) -> Dict:
        """Generate header part of the transformed code."""
        _instruction = (self._do_fullcode_instruction(element) if is_full 
                       else self._do_fragcode_header(element))
        parts = {
            "instruction": _instruction,
            "signature": f" with signature {self._restore_cpp_function(element)}",
            "location": f" located at file: {element["definition"]['location']}, line:{element["definition"]['line']}"
        }
        if element["definition"]["context_before"]:
            parts["context_before"] = f"\nContext before is: \"{'\n'.join(element["definition"]['context_before'])}\""
        parts.update(self._add_doc_and_comment(element))
        return self._build_truncated_input(parts)
    
    def _body(self, element: Dict, frag: int, is_end: bool = True) -> Dict:
        """Generate body part of the transformed code."""
        _instruction = (self._do_fragcode_tail(element) if is_end 
                       else self._do_fragcode_body(str(frag), element))
        parts = {
            "instruction":_instruction,
            "signature":f" with signature {self._restore_cpp_function(element)}",
            "location":f" located at file {element["definition"]['location']}"
        }
        parts.update(self._add_doc_and_comment(element))     
        return self._build_truncated_input(parts)
    
    def _docstring_generate(self, element: Dict):
        _doc = self._get_docstring(element)
        if _doc:
            _inst = self._do_docsting_instruction(element)
            _code = element["definition"]["code"]
            _inst += f" with code: {_code}"
            self.documenter.append({"output": _doc, "input": _inst})

    def _restore_cpp_function(self, element: Dict) -> str:
        if element["declaration"] is not None:
            return element["declaration"]["code"]
        else:    
            return generate_signature(element["definition"])
    
    def transform(self, element: Dict) -> Dict:
        """Transform a single function element."""
        if (element["definition"] is None) or (element["def_cnt"] > 1) or (element["decl_cnt"] > 1):
            self.skipped_counter += 1
            return
        element["definition"]["code"] = self.clean_cpp_code(element["definition"]["code"])
        self._docstring_generate(element)
        self._process_code_fragments(element)


class ClassTransformer(DsTransformerBase):
    """Transformer for class/struct elements that only have definitions."""
    
    def __init__(self, max_tokens: int = 512, 
                 tkn: Optional[DsTokenCounter] = None, 
                 fragmenter: Optional[CodeFragmenter] = None,
                 deration: float = 0.75):
        super().__init__(max_tokens, tkn, fragmenter)
        self.code_writer = []
        self.documenter = []
    
    def generate_full_name(self, element):
        name = element["type"] + "_" + element["name"]
        if element["parent_type"] != "translation_unit":
            name = element["parent_type"] + "_" + element["parent_name"] + "_" + name
        return name
    
    def transform(self, element: Dict) -> Dict:
        """Transform a single class element."""
        if element["def_cnt"] > 1:
            self.skipped_counter += 1
            return
        element["definition"]["code"] = self.clean_cpp_code(element["definition"]["code"])
        self._docstring_generate(element)
        self._process_code_fragments(element)


class DatasetTransformer():
    def __init__(self, input_path, repo_name, token_counter: Optional[DsTokenCounter] = None):
        self.dstkn = token_counter
        self.frag = CodeFragmenter()
        self.ft = FunctionsTransformer(tkn = self.dstkn, fragmenter=self.frag, deration = 0.6)
        self.cl = ClassTransformer(tkn = self.dstkn, fragmenter=self.frag, deration = 0.8)
        self.enumvar = ClassTransformer(tkn = self.dstkn, fragmenter=self.frag, deration = 0.8)
        self.input_file = os.path.join(input_path, f"dataset_clang_{repo_name}.jsonl")
        self.output_file_code_ft = os.path.join(input_path, f"dataset_code_finetune_{repo_name}.jsonl")
        self.output_file_doc_ft = os.path.join(input_path, f"dataset_doc_finetune_{repo_name}.jsonl")
        self.functions = {}
        self.classes = {}
        self.enumvars = {}

    def _extract_function(self, entry):
        name = self.ft.generate_full_name(entry)
        # print(f"function name:\t{name}")
        if name not in self.functions.keys():
            self.functions[name] = {"definition": None, "declaration": None, "def_cnt": 0, "decl_cnt": 0}
        if entry["is_defined"]:
            self.functions[name]["def_cnt"] += 1
            if self.functions[name]["definition"] is None:
                self.functions[name]["definition"] = entry
            # else:
                # print(f"func {name} \"definition\" is not None!\t{entry["location"]}:{entry["line"]}\t||\t{self.functions[name]["definition"]["name"]}\t{self.functions[name]["definition"]["location"]}:{self.functions[name]["definition"]["line"]}")
        else:
            self.functions[name]["decl_cnt"] += 1
            if self.functions[name]["declaration"] is None:
                self.functions[name]["declaration"] = entry
            # else:
                # print(f"func {name} \"declaration\" is not None!\t{entry["location"]}:{entry["line"]}\t||\t{self.functions[name]["declaration"]["name"]}\t{self.functions[name]["declaration"]["location"]}:{self.functions[name]["declaration"]["line"]}")

    def _extract_class(self, entry):
        name = self.cl.generate_full_name(entry)
        # print(f"class name:\t{name}")
        if name not in self.classes.keys():
            self.classes[name] = {"definition": entry, "def_cnt": 1}
        else:
            self.classes[name]["def_cnt"] += 1
            # print(f"class {name} defiend Again!\t{entry["location"]}:{entry["line"]}\t||\t{self.classes[name]["definition"]["name"]}\t{self.classes[name]["definition"]["location"]}:{self.classes[name]["definition"]["line"]}")

    def _extract_enum_and_vars(self, entry):
        name = self.enumvar.generate_full_name(entry)
        if name not in self.enumvars.keys():
            self.enumvars[name] = {"definition": entry, "def_cnt": 1}
        else:
            self.enumvars[name]["def_cnt"] += 1
            print(f"enumvars {name} defiend Again!\t{entry["location"]}:{entry["line"]}\t||\t{self.enumvars[name]["definition"]["name"]}\t{self.enumvars[name]["definition"]["location"]}:{self.enumvars[name]["definition"]["line"]}")

    def prepare_lists(self):
        with open(self.input_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                entry = json.loads(line)
                if entry['type'] in ("cxx_method", "constructor", "destructor", "function_decl", "function_template"):
                    self._extract_function(entry)
                elif entry['type'] in ("class_decl", "struct_decl", "class_template", "class_template_partial_specialization"):
                    self._extract_class(entry)
                elif entry['type'] in ("var_decl", "enum_decl"):
                    self._extract_enum_and_vars(entry)

    def parse_functions(self):
        for _, func in self.functions.items():
            self.ft.transform(func)
        print(f" self.ft.code_writer is {len(self.ft.code_writer)}")
        self._stat_subprocess(self.ft.code_writer)
        print(f" self.ft.documenter length is {len(self.ft.documenter)}")
        self._stat_subprocess(self.ft.documenter)
        print(f"too long inputs: {self.ft.too_long_inp_counter}\t skipped functions: {self.ft.skipped_counter}")

    def parse_class(self):
        for _, cls in self.classes.items():
            self.cl.transform(cls)
        print(f" self.cl.code_writer is {len(self.cl.code_writer)}")
        self._stat_subprocess(self.cl.code_writer)
        print(f" self.cl.documenter length is {len(self.cl.documenter)}")
        self._stat_subprocess(self.cl.documenter)
        print(f"too long inputs: {self.cl.too_long_inp_counter}\t skipped classes: {self.cl.skipped_counter}")
    
    def parse_enumvars(self):
        for _, el in self.enumvars.items():
            self.enumvar.transform(el)
        print(f" self.enumvar.code_writer is {len(self.enumvar.code_writer)}")
        self._stat_subprocess(self.enumvar.code_writer)
        print(f" self.enumvar.documenter length is {len(self.enumvar.documenter)}")
        self._stat_subprocess(self.enumvar.documenter)
        print(f"too long inputs: {self.enumvar.too_long_inp_counter}\t skipped enum & vars: {self.enumvar.skipped_counter}")
    
    def save_data(self):
        with open(self.output_file_code_ft, "w", encoding="utf-8") as f:
            for element in self.ft.code_writer:
                json_line = json.dumps(element, ensure_ascii=False)
                f.write(json_line + "\n")
            for element in self.cl.code_writer:
                json_line = json.dumps(element, ensure_ascii=False)
                f.write(json_line + "\n")
            for element in self.enumvar.code_writer:
                json_line = json.dumps(element, ensure_ascii=False)
                f.write(json_line + "\n")
        with open(self.output_file_doc_ft, "w", encoding="utf-8") as f:
            for element in self.ft.documenter:
                json_line = json.dumps(element, ensure_ascii=False)
                f.write(json_line + "\n")
            for element in self.cl.documenter:
                json_line = json.dumps(element, ensure_ascii=False)
                f.write(json_line + "\n")
            for element in self.enumvar.documenter:
                json_line = json.dumps(element, ensure_ascii=False)
                f.write(json_line + "\n")

    def review_lists(self):
        _cls_multi_definition = 0
        for name, cls in self.classes.items():
            if cls["def_cnt"] > 1:
                _cls_multi_definition += 1
                print(f"multi defined\t{name} in {cls["definition"]["location"]}:{cls["definition"]["line"]}")
        _undefined = 0
        _multi_definition = 0
        _multi_declaration = 0
        print(f"functions count is {len(self.functions)}")
        print("definition:declaration")
        for name, func in self.functions.items():
            if func["definition"] is None:
                _undefined += 1
                print(f"not defined\t{func["definition"] is not None}:{func["declaration"] is not None}\t{name}\t{func["declaration"]["location"]}:{func["declaration"]["line"]}")
            elif func["def_cnt"] > 1:
                _multi_definition += 1
                print(f"multi defined\t{func["definition"] is not None}:{func["declaration"] is not None}\t{name}")
            elif func["decl_cnt"] > 1:
                _multi_declaration += 1
                print(f"multi declared\t{func["definition"] is not None}:{func["declaration"] is not None}\t{name}")
        print(f"class count is {len(self.classes)}\tclass multi definition count is {_cls_multi_definition}")
        print(f"functions count is {len(self.functions)}\tnot defined: {_undefined}\tmulti deifned: {_multi_definition}\tmulti declared: {_multi_declaration}")

    def plot_single_histograms(self, name, lengths):
        """Функция для построения гистограмм длин кода по типам элементов"""
        # Пропускаем служебные поля
                    
        plt.figure(figsize=(12, 7))
        n_bins = 50
        _, bins, _ = plt.hist(lengths, bins=n_bins, alpha=0.7, color='blue', log=True)
        
        # Вычисляем размер бина (берем первый интервал как пример)
        bin_size = bins[1] - bins[0]
        plt.title(f'{name}\n'
                f'Total elements: {len(lengths)}, Bin size: {bin_size:.1f} chars')
        plt.xlabel('Code length (characters)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        # Добавляем вертикальные линии с полным описанием
        plt.axvline(x=lengths.min(), color='red', linestyle='--', 
                label=f'Min: {lengths.min()}')
        plt.axvline(x=lengths.max(), color='green', linestyle='--', 
                label=f'Max: {lengths.max()}')
        # Настраиваем легенду с переносом текста
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def _stat_subprocess(self, dataset):
        _lens = np.zeros((2, len(dataset)))
        for _idx, el in enumerate(dataset):
            _lens[0][_idx] = self.dstkn.get_token_count(el["input"])
            _lens[1][_idx] = self.dstkn.get_token_count(el["output"])

        # for idx in range(10):
        #     print(dataset[idx], "\n\n")
        # for i in range(2):
            # print(["inputs", "outputs"][i], f"mean: {_lens[i].mean()} min: {_lens[i].min()} max: {_lens[i].max()}")
        
        # min_idx = _lens[0].argmin()
        # max_idx = _lens[0].argmax()
        # print(f"min input is {dataset[min_idx]}")
        # print(f"max input is {dataset[max_idx]}")

        # min_idx = _lens[1].argmin()
        # max_idx = _lens[1].argmax()
        # print(f"min output is {dataset[min_idx]}")
        # print(f"max output is {dataset[max_idx]}")

        if len(dataset) > 0: 
            self.plot_single_histograms("input", _lens[0])
            self.plot_single_histograms("output", _lens[1])


if __name__ == "__main__":

    signature = {"name": "string_dup", "return_type": "char *", "parameters": [{"name": "s", "type": "const std::string &", "default_value": None}], 
                 "qualifiers": {"is_const": False, "is_static": False, "is_virtual": False, "is_pure_virtual": False, "is_noexcept": False}}
    # element = {"id": "FUNCT_dacd09907bf9", "type": "function_decl", "name": "main", 
    #            "signature": {"name": "main", "return_type": "int", 
    #                          "parameters": [{"name": "param_0", "type": "int", "default_value": None}, 
    #                                         {"name": "param_1", "type": "char **", "default_value": None}],
    #                          "qualifiers": {"is_const": False, "is_static": False, "is_virtual": False, "is_pure_virtual": False, "is_noexcept": False}}, 
    #             "code": "int main(int, char**)\n{\n  zmq::context_t c;\n  zmq::socket_t s(c, ZMQ_REQ);\n  s.disconnect(\"some endpoint\");\n}", 
    #             "location": "cppTango-9.3.7\\configure\\test_cppzmq_disconnect.cpp", 
    #             "line": 3, 
    #             "parent_id": "TRANS_ca90eeafe571", 
    #             "parent_type": "TRANSLATION_UNIT", 
    #             "parent_name": "C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\configure\\test_cppzmq_disconnect.cpp", 
    #             "is_defined": "True", 
    #             "context_before": ["#include <zmq.hpp>", ""], 
    #             "context_after": ["int main(int, char**)", "{", "zmq::context_t c;"], 
    #             "comments": [], 
    #             "docstrings": []}
    
    element = {"id": "CONST_c6a4f8b73ee1", "type": "constructor", "name": "ApiUtil", "parent_id": "CLASS_a03bbf628416", "parent_type": "CLASS_DECL", "parent_name": "ApiUtil", "signature": {"name": "ApiUtil", "return_type": "void", "parameters": [], "qualifiers": {"is_const": False, "is_static": False, "is_virtual": False, "is_pure_virtual": False, "is_noexcept": False}}, "code": "ApiUtil();", "is_defined": "False", "location": "cppTango-9.3.7\\cppapi\\client\\ApiUtil.h", "line": 213, "is_template": False, "context_before": ["", "protected:", "/// @privatesection"], "context_after": ["ApiUtil();", "virtual ~ApiUtil();", ""], "comments": [], "docstrings": [{"type": "line", "text": "/// @privatesection\n", "line": 212}]}

    dstkn = DsTokenCounter()
    # from code_fragmenter import get_code_element
    # from pathlib import Path
    # _code_element = get_code_element(_type="constructor", _name = "DeviceAttribute", _file = Path(f"C:\\work\\llm_test\\dataset_clang_cppTango-9.3.7.jsonl"))
    # print(len(_code_element))
    # ft = CxxMethodTransformer(tkn = dstkn, method_type="constructor")
    # print(ft.transform(_code_element))
    # # print(ft.transform(element))
    # # print(ft._restore_cpp_function(signature))
    # REPO_NAME = "template_exampl"
    REPO_NAME = "cppTango-9.3.7"
# C:\work\pavlenko\llmtest-git\dataset_clang_cppTango-9.3.7.jsonl
# C:\\work\\pavlenko\\llmtest-git
    # dt = DatasetTransformer(input_path=f"C:\\work\\llm_test", repo_name=REPO_NAME)
    # dt = DatasetTransformer(input_path=f"C:\\work\\llm_test", repo_name=REPO_NAME, token_counter=dstkn)
    dt = DatasetTransformer(input_path=f"C:\\work\\pavlenko\\llmtest-git", repo_name=REPO_NAME, token_counter=dstkn)
    dt.prepare_lists()
    dt.parse_functions()
    dt.parse_class()
    dt.parse_enumvars()
    dt.save_data()