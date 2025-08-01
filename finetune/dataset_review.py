import json
import os
from pathlib import Path
import re
from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling,
#     BitsAndBytesConfig
# )



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
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_cppTango-9.3.7.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_simple.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_adc4x250.jsonl")
INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_template_exampl.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_overload_example.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\pavlenko\\llmtest-git\\dataset_clang_test_examples.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\pavlenko\\llmtest-git\\dataset_clang_template_test.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\pavlenko\\llmtest-git\\dataset_clang_template_test_simple.jsonl")


def get_function(_type = "classes", _name = "DeviceImpl", _file = INPUT_JSONL):
    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:

        for line in in_f:
            entry = json.loads(line)
            # print(len(entry), entry['type'])
            if entry['type'] == "classes" and entry["name"] == "DeviceImpl":
                
                _desc = f"{entry['type']}:\t{entry["name"]} with type {entry["type"]}"
                _doc = entry.get("docstring", "")
                _code = entry.get("code", "")
                _sgntr = entry.get("signature", "")
                _body =  entry.get("full_body", "")
                _is_defined = entry.get("is_defined", "None")
                _comment = "".join([el["text"] for el in entry["comments"]])
                _docstring = "".join([el["text"] for el in entry["docstrings"]])

                print(f"{_desc}")
                print("comment is\n", _comment)
                print("docstring is", _docstring)
                # print(_doc)
                # print(_sgntr)
                # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, {[len(x) for x in _code.split("\n")]}")
                # print(f"body: {_body}")
                return _code
    return ""

def show_elements(_file = INPUT_JSONL):
    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:

        for line in in_f:
            entry = json.loads(line)
            # print(len(entry), entry['type'])
            # if entry['type'] == "function_template":
            if True:     
                _code = entry.get("code", "")
                _is_defined = entry.get("is_defined", "None")
                _desc = f"{entry['type']}:\t{entry["name"]}\twith type {entry["type"]}\tparent nema: {entry["parent_name"]}\tparent type: {entry["parent_type"]}\tat {entry["location"]}:{entry["line"]}\tcode length {len(_code)}\tis_defined: {_is_defined}"
                _doc = entry.get("docstring", "")
                _sgntr = entry.get("signature", "")
                _body =  entry.get("full_body", "")
                _comment = "".join([el["text"] for el in entry["comments"]])
                _docstring = "".join([el["text"] for el in entry["docstrings"]])

                print(f"{_desc}")
                # print("comment is\n", _comment)
                # print("docstring is", _docstring)
                # print(_doc)
                # print(_sgntr)
                # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, {[len(x) for x in _code.split("\n")]}")
                # print(f"body: {_body}")
                # return _code
    # return ""
def _get_parameters(signature):
    parameters = []
    for param in signature["parameters"]:
        param_str = f"{param['type']}"
        if param["name"]:
            param_str += f"_{param['name']}"
        if param["default_value"] is not None:
            param_str += f"[{param['default_value']}]"
        parameters.append(param_str)
    return parameters

def generate_func_full_name(element):
    _name = element["type"] + "_" + element["name"]
    if element["parent_type"] != "translation_unit":
        _name = element["parent_type"] + "_" + element["parent_name"] + "_" + _name
    signature = element["signature"]
    _name += f"_ret_{signature["return_type"]}"
    parameters = _get_parameters(signature)
    if len(parameters) != 0:
        _name += '_'
        _name += '_'.join(parameters)
    return _name

def get_functions_set(_file = INPUT_JSONL):
    functions = {}
    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:
        for line in in_f:
            entry = json.loads(line)
            if entry['type'] in (
                "cxx_method",
                "constructor",
                "destructor",
                "function_decl",
                "function_template"
            ):
                f_name = generate_func_full_name(entry)
                print(f_name)
                if f_name not in functions.keys():
                    functions[f_name] = {"definition": None, "declaration": None, "def_cnt": 0, "decl_cnt": 0}
                if entry["is_defined"]:
                    functions[f_name]["def_cnt"] += 1
                    if functions[f_name]["definition"] is not None:
                        print(f"{functions[f_name]}[\"definition\"] is not None!")
                    else:
                        functions[f_name]["definition"] = entry
                else:
                    functions[f_name]["decl_cnt"] += 1
                    if functions[f_name]["declaration"] is not None:
                        print(f"{functions[f_name]}[\"declaration\"] is not None!")
                    else:
                        functions[f_name]["declaration"] = entry
    return functions

def is_not_defined(function):
    return function["definition"] is None

def is_method(function):
    return function["definition"]["parent_type"] in (
        "class_decl",
        "struct_decl",
        "class_template",
        "class_template_partial_specialization")

def separate_function_declaration_and_body(code_str):
    """
    Разделяет объявление функции и её тело.
    Предполагается, что комментарии уже удалены.
    
    Args:
        code_str (str): Строка с кодом функции/метода без комментариев.
        
    Returns:
        tuple: (declaration, body) - объявление и тело функции
               Если тело не найдено, body будет пустой строкой.
    """
    # Ищем первую фигурную скобку
    brace_pos = code_str.find('{')
    
    if brace_pos != -1:
        declaration = code_str[:brace_pos].strip()
        body = code_str[brace_pos:].strip()
        return declaration, body
    
    # Ищем точку с запятой в конце (для однострочных функций)
    if code_str.rstrip().endswith(';'):
        declaration = code_str.strip()
        return declaration, ""
    
    # Если ничего не найдено, вернем весь код как объявление
    return None, None




if __name__ == "__main__":
        # Загрузка токенизатора
    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    # _code = get_function()
    # input_tokens_no_trunc = tokenizer(_code, truncation=False)

    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(_code)/len(input_tokens_no_trunc[0])}")

    # _code = clean_cpp_code(_code)
    # input_tokens_no_trunc = tokenizer(_code, truncation=False)
    # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, tokens = {len(input_tokens_no_trunc[0])}, ratio = {len(_code)/len(input_tokens_no_trunc[0])}")
    # fncs = get_functions_set()
    # print(len(fncs))
    # for nm, fn in fncs.items():
    #     print(f"definition: {fn["definition"] is not None}\tdeclaration: {fn["declaration"] is not None}\t\t{nm}")
    # Пример использования
    code = "template <typename T>\nvoid ClassC::myMethod(T value) {\n    std::cout << value << std::endl;\n}"

    declaration, body = separate_function_declaration_and_body(code)
    print("Declaration:")
    print(declaration)
    print("\nBody:")
    print(body)