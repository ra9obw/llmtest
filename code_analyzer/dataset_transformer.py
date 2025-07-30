from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer
    )
from dataclasses import dataclass

@dataclass
class DsTokenCounterConfig:
    MODEL_NAME = "Qwen/Qwen3-0.6B"


class DsTokenCounter:
    def __init__(self, config: DsTokenCounterConfig = DsTokenCounterConfig()):
        self._config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self._config.MODEL_NAME, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Нужно для padding
    
    def get_token_count(self, in_str: str) -> int:
        _input_tokens_no_trunc = self.tokenizer(in_str, truncation=False)
        _len = len(_input_tokens_no_trunc[0])
        return _len
    
    def get_batch_token_count(self, in_str: List[str]) -> List[int]:
        _input_tokens_no_trunc = self.tokenizer(in_str, truncation=False)
        # print((_input_tokens_no_trunc))
        _len = [len(_inp) for _inp in _input_tokens_no_trunc['input_ids']]
        return _len


class DsTransformerBase:
    def __init__(self, max_tokens: int = 512, tkn: DsTokenCounter = None):
        self.max_tokens = max_tokens
        self._tkn = tkn

    def restore_cpp_function(self, signature):
        # Собираем квалификаторы
        qualifiers = []
        if signature["qualifiers"]["is_static"]:
            qualifiers.append("static")
        if signature["qualifiers"]["is_virtual"]:
            qualifiers.append("virtual")
        if signature["qualifiers"]["is_const"]:
            qualifiers.append("const")
        if signature["qualifiers"]["is_noexcept"]:
            qualifiers.append("noexcept")
        if signature["qualifiers"]["is_pure_virtual"]:
            qualifiers.append("= 0")
        
        # Собираем параметры
        parameters = []
        for param in signature["parameters"]:
            param_str = f"{param['type']}"
            if param["name"]:
                param_str += f" {param['name']}"
            if param["default_value"] is not None:
                param_str += f" = {param['default_value']}"
            parameters.append(param_str)
        
        # Собираем полную сигнатуру
        parts = []
        if qualifiers and not signature["qualifiers"]["is_pure_virtual"]:
            parts.extend(qualifiers[:-1])  # Все кроме последнего (который может быть "= 0")
        
        parts.append(signature["return_type"])
        parts.append(signature["name"] + "(" + ", ".join(parameters) + ")")
        
        if signature["qualifiers"]["is_const"] and "const" in qualifiers:
            parts.append("const")
        if signature["qualifiers"]["is_noexcept"] and "noexcept" in qualifiers:
            parts.append("noexcept")
        if signature["qualifiers"]["is_pure_virtual"] and "= 0" in qualifiers:
            parts.append("= 0")
        
        return " ".join(parts)

    def transform(self, element: Dict) -> Dict:
        pass


class FunctionTransformer(DsTransformerBase):
    def __init__(self, max_tokens: int = 512, tkn: DsTokenCounter = None):
        super().__init__(max_tokens, tkn)
    
    def _header(self, element: Dict, code_snippet: str):
        _instruction = f"Implement C++ function {element["name"]}"
        _signature = f" with signature {ft.restore_cpp_function(signature)}"
        _location = f" located at file {element["location"]}:{element["line"]}"
        if len(element["context_before"]) != 0:
            _context = f"\nContext before is: \"{"\n".join(element["context_before"])}\""
        else:
            _context = None
        if len(element["docstrings"]) != 0:
            _docstring = f" using docstring {"\n".join(element["docstrings"])}"
        else:
            _docstring = None
        if len(element["comments"]) != 0:
            _comment = f" using comments {"\n".join(element["comments"])}"
        else:
            _comment = None
        input = [_instruction, _signature, _location]
        if _context: input.append(_context)
        if _docstring: input.append(_docstring)
        if _comment: input.append(_comment)
        _str = "".join(input)
        tkn_count = self._tkn.get_batch_token_count(input)
        print(tkn_count, sum(tkn_count))
        print(self._tkn.get_token_count(_str))
        return {"input": _str, "output": code_snippet}
    
    def transform(self, element: Dict) -> Dict:
        _ret = {"code_writing": [], "doc_generation": []}
        if element["is_defined"] == "True":
            if (self._tkn == None) or self._tkn.get_token_count(element["code"]) <= self.max_tokens:
                _ret["code_writing"].append(self._header(element, element["code"]))
            else:
                
        return _ret



if __name__ == "__main__":
    signature = {"name": "string_dup", "return_type": "char *", "parameters": [{"name": "s", "type": "const std::string &", "default_value": None}], 
                 "qualifiers": {"is_const": False, "is_static": False, "is_virtual": False, "is_pure_virtual": False, "is_noexcept": False}}
    element = {"id": "FUNCT_dacd09907bf9", "type": "function_decl", "name": "main", 
               "signature": {"name": "main", "return_type": "int", 
                             "parameters": [{"name": "param_0", "type": "int", "default_value": None}, 
                                            {"name": "param_1", "type": "char **", "default_value": None}],
                             "qualifiers": {"is_const": False, "is_static": False, "is_virtual": False, "is_pure_virtual": False, "is_noexcept": False}}, 
                "code": "int main(int, char**)\n{\n  zmq::context_t c;\n  zmq::socket_t s(c, ZMQ_REQ);\n  s.disconnect(\"some endpoint\");\n}", 
                "location": "cppTango-9.3.7\\configure\\test_cppzmq_disconnect.cpp", 
                "line": 3, 
                "parent_id": "TRANS_ca90eeafe571", 
                "parent_type": "TRANSLATION_UNIT", 
                "parent_name": "C:\\work\\pavlenko\\llmtest-git\\codebase\\cppTango-9.3.7\\cppTango-9.3.7\\configure\\test_cppzmq_disconnect.cpp", 
                "is_defined": "True", 
                "context_before": ["#include <zmq.hpp>", ""], 
                "context_after": ["int main(int, char**)", "{", "zmq::context_t c;"], 
                "comments": [], 
                "docstrings": []}
    dstkn = DsTokenCounter()
    ft = FunctionTransformer(tkn = dstkn)
    print(ft.transform(element))
    # print(ft.restore_cpp_function(signature))