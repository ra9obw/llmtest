import json
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
                _desc = f"{entry['type']}:\t{entry["name"]}\twith type {entry["type"]}\tparent nema: {entry["parent_name"]}\tparent type: {entry["parent_type"]}\tat {entry["location"]}:{entry["line"]}\t\tcode length {len(_code)}\tis_defined: {_is_defined}"
                _doc = entry.get("docstring", "")
                _sgntr = entry.get("signature", "")
                _body =  entry.get("full_body", "")
                _comment = "".join([el["text"] for el in entry["comments"]])
                _docstring = "".join([el["text"] for el in entry["docstrings"]])

                print(f"{_desc}")
                print("comment is\n", _comment)
                print("docstring is", _docstring)
                # print(_doc)
                # print(_sgntr)
                # print(f"code length {len(_code)}, lines count {len(_code.split("\n"))}, {[len(x) for x in _code.split("\n")]}")
                # print(f"body: {_body}")
                # return _code
    # return ""

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
    show_elements()