import json
from pathlib import Path

# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_template_exampl.jsonl")
INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_overload_example.jsonl")

with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:

    for line in in_f:
        entry = json.loads(line)
        print(len(entry), entry['type'])
        if entry['type'] == "class" or entry['type'] == "class_template":
            for method in entry.get("methods", []):
                _desc = f"{entry['type']} '{entry['name']}' : {method['type']} {method['name']}"
                _doc = method.get("docstring", "")
                _code = method.get("code", "")
                _sgntr = method.get("signature", "")
                _body =  method.get("full_body", "")
                print(_desc)
                # print(_doc)
                # print(_sgntr)
                print(f"code: {_code}")
                print(f"body: {_body}")
