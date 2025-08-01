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
        fragmenter: Optional[CodeFragmenter] = None
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._tkn = tkn
        self.fragmenter = fragmenter

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
    
    def transform(self, element: Dict) -> Dict:
        """Transform the input element."""
        raise NotImplementedError


class FunctionsTransformer(DsTransformerBase):
    """Generic code transformer that can handle different element types."""
    
    def __init__(self, max_tokens: int = 512, tkn: Optional[DsTokenCounter] = None):
        super().__init__(max_tokens, tkn)
    
    def _do_fullcode_instruction(self, element: Dict) -> str:
        return generate_instruction(element, GenerationMode.FULL)
    
    def _do_fragcode_header(self, element: Dict) -> str:
        return generate_instruction(element, GenerationMode.FIRST_FRAGMENT)

    def _do_fragcode_body(self, idx: str, element: Dict) -> str:
        return generate_instruction(element, GenerationMode.NEXT_FRAGMENT, idx)

    def _do_fragcode_tail(self, element: Dict) -> str:
        return generate_instruction(element, GenerationMode.LAST_FRAGMENT)    

    def _get_parameters(self, signature):
        parameters = []
        for param in signature["parameters"]:
            param_str = f"{param['type']}"
            if param["name"]:
                param_str += f"_{param['name']}"
            if param["default_value"] is not None:
                param_str += f"[{param['default_value']}]"
            parameters.append(param_str)
        return parameters

    def generate_full_name(self, element):
        _name = element["type"] + "_" + element["name"]
        if element["parent_type"] != "translation_unit":
            _name = element["parent_type"] + "_" + element["parent_name"] + "_" + _name
        signature = element["signature"]
        _name += f"_ret_{signature["return_type"]}"
        parameters = self._get_parameters(signature)
        if len(parameters) != 0:
            _name += '_'
            _name += '_'.join(parameters)
        return _name

    def _header(self, element: Dict, code_snippet: str, is_full: bool = True) -> Dict:
        """Generate header part of the transformed code."""
        _instruction = (self._do_fullcode_instruction(element) if is_full 
                       else self._do_fragcode_header(element))
        parts = [
            _instruction,
            f" with signature {self._restore_cpp_function(element)}",
            f" located at file: {element['location']}, line:{element['line']}"
        ]
        
        if element["context_before"]:
            parts.append(f"\nContext before is: \"{'\n'.join(element['context_before'])}\"")
        if element["docstrings"]:
            parts.append(f" using docstring {'\n'.join(element['docstrings'])}")
        if element["comments"]:
            parts.append(f" using comments {''.join(el['text'] for el in element['comments'])}")
            
        return {"input": "".join(parts), "output": code_snippet}
    
    def _body(self, element: Dict, code_snippet: str, frag: int, is_end: bool = True) -> Dict:
        """Generate body part of the transformed code."""
        _instruction = (self._do_fragcode_tail(element) if is_end 
                       else self._do_fragcode_body(str(frag), element))
        parts = [
            _instruction,
            f" with signature {self._restore_cpp_function(element)}",
            f" located at file {element['location']}"
        ]
        
        if element["docstrings"]:
            parts.append(f" using docstring {'\n'.join(element['docstrings'])}")
        if element["comments"]:
            parts.append(f" using comments {'\n'.join(element['comments'])}")
            
        return {"input": "".join(parts), "output": code_snippet}
    
    def _restore_cpp_function(self, element: Dict) -> str:
        return generate_signature(element)

    def _func_transform(self, element: Dict) -> Dict:
        """Transform a single function element."""
        _ret = {"code_writing": [], "doc_generation": []}
        
        if element.get("is_defined") != "True":
            return _ret
            
        if self._tkn and self.fragmenter:
            _tkn_count = self._tkn.get_token_count(element["code"])
            
        if (not self._tkn or not self.fragmenter) or _tkn_count <= self.max_tokens:
            _ret["code_writing"].append(self._header(element, element["code"]))
        else:
            self.fragmenter.config.chars_per_token = 0.9 * _tkn_count / len(element["code"])
            frags = self.fragmenter.split_code(element)
            
            for _idx, frag in enumerate(frags):
                if _idx == 0:
                    _ret["code_writing"].append(
                        self._header(element, frag["code"], is_full=False)
                    )
                else:
                    _ret["code_writing"].append(
                        self._body(element, frag["code"], _idx, is_end=(_idx == len(frags)-1))
                    )
        
        return _ret
    
    def transform(self, element: Dict) -> Dict:
        return self._func_transform(element)
    

class ClassTransformer(DsTransformerBase):

    def __init__(self, max_tokens: int = 512, tkn: Optional[DsTokenCounter] = None):
        super().__init__(max_tokens, tkn)

    def generate_full_name(self, element):
        name = element["type"] + "_" + element["name"]
        if element["parent_type"] != "translation_unit":
            name = element["parent_type"] + "_" + element["parent_name"] + "_" + name
        return name

class DatasetTransformer():
    def __init__(self, input_path, repo_name):
        self.dstkn = DsTokenCounter()
        self.ft = FunctionsTransformer(tkn = self.dstkn)
        self.cl = ClassTransformer(tkn = self.dstkn)
        self.input_file = os.path.join(input_path, f"dataset_clang_{repo_name}.jsonl")
        self.output_file_code_ft = os.path.join(input_path, f"dataset_code_finetune_{repo_name}.jsonl")
        self.output_file_doc_ft = os.path.join(input_path, f"dataset_doc_finetune_{repo_name}.jsonl")
        self.functions = {}
        self.classes = {}

    def _extract_function(self, entry):
        name = self.ft.generate_full_name(entry)
        # print(f"function name:\t{name}")
        if name not in self.functions.keys():
            self.functions[name] = {"definition": None, "declaration": None, "def_cnt": 0, "decl_cnt": 0}
        if entry["is_defined"]:
            self.functions[name]["def_cnt"] += 1
            if self.functions[name]["definition"] is not None:
                print(f"func {name} \"definition\" is not None!\t{entry["location"]}:{entry["line"]}")
            else:
                self.functions[name]["definition"] = entry
        else:
            self.functions[name]["decl_cnt"] += 1
            if self.functions[name]["declaration"] is not None:
                print(f"func {name} \"declaration\" is not None!\t{entry["location"]}:{entry["line"]}")
            else:
                self.functions[name]["declaration"] = entry

    def _extract_class(self, entry):
        name = self.cl.generate_full_name(entry)
        # print(f"class name:\t{name}")
        if name not in self.classes.keys():
            self.classes[name] = {"definition": entry, "def_cnt": 1}
        else:
            self.classes[name]["def_cnt"] += 1
            print(f"class {name} defiend Again!\t{entry["location"]}:{entry["line"]}")

    def prepare_lists(self):
        with open(self.input_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                entry = json.loads(line)
                if entry['type'] in ("cxx_method", "constructor", "destructor", "function_decl", "function_template"):
                    self._extract_function(entry)
                elif entry['type'] in ("class_decl", "struct_decl", "class_template", "class_template_partial_specialization"):
                    self._extract_class(entry)
        print(f"functions count: {len(self.functions)};\tclass count: {len(self.classes)}")

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

    # dstkn = DsTokenCounter()
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

    dt = DatasetTransformer(input_path=f"C:\\work\\llm_test", repo_name=REPO_NAME)
    dt.prepare_lists()