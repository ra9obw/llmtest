import json
from pathlib import Path


# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_cppTango-9.3.7.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_simple.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_adc4x250.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_template_exampl.jsonl")
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_overload_example.jsonl")
INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_test_examples.jsonl")

with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:

    for line in in_f:
        entry = json.loads(line)
        # print(len(entry), entry['type'])
        # if entry['type'] == "class" or entry['type'] == "class_template":
            
        _desc = f"{entry['type']}:\t{entry["data"]["name"]} with type {entry["data"]["type"]}"
        _doc = entry.get("docstring", "")
        _code = entry["data"].get("code", "")
        _sgntr = entry.get("signature", "")
        _body =  entry["data"].get("full_body", "")
        _is_defined = entry["data"].get("is_defined", "None")
        _comment = "".join([el["text"] for el in entry["data"]["comments"]])
        _docstring = "".join([el["text"] for el in entry["data"]["docstrings"]])

        print(f"{_desc}")
        # print(_comment)
        # print(_docstring)
        # print(_doc)
        # print(_sgntr)
        print(f"code: {_code}")
        # print(f"body: {_body}")
