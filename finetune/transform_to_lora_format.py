import json
from pathlib import Path

# INPUT_JSONL = Path("C:\\work\\llm_test\\dataset_clang_adc4x250.jsonl")
# INPUT_JSONL = Path("C:\\work\\llm_test\\dataset_clang_adc4x250_doc.jsonl")
prefix = "doc"
# INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_adc4x250_{prefix}.jsonl")
INPUT_JSONL = Path(f"C:\\work\\llm_test\\dataset_clang_adc4x250_{prefix}.jsonl")
OUTPUT_PATH = Path(f"C:\\work\\llm_test\\{prefix}.txt")

with open(INPUT_JSONL, "r", encoding="utf-8") as in_f, open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:

    for line in in_f:
        entry = json.loads(line)
        print(len(entry), entry['type'])
        if entry['type'] == "class":
            for method in entry.get("methods", []):
                desc = f"Class '{entry['name']}' : method {method['name']}"
                doc = method.get("docstring", "")
                print(desc)
                print(doc)
                out_f.write(desc)
                out_f.write(doc)
                out_f.write("\n")

        # Обработка функций
        for func in entry.get("functions", []):
            print(func['name'])
            # if "code" in func:
            #     input_text = f"Implement function '{func['name']}'"
            #     output_text = func["code"]
            #     out_f.write(json.dumps({"input": input_text, "output": output_text}, ensure_ascii=False) + "\n")

        # Обработка классов и их методов
        for cls in entry.get("classes", []):
            print(f"Class '{cls['name']}'")
            # if "code" in cls:
            #     input_text = f"Implement class '{cls['name']}'"
            #     output_text = cls["code"]
            #     out_f.write(json.dumps({"input": input_text, "output": output_text}, ensure_ascii=False) + "\n")

            for method in cls.get("methods", []):
                print(f"Class '{cls['name']}' : method {method['name']}")
                # if "code" in method:
                #     input_text = f"Implement method '{method['name']}' in class '{cls['name']}'"
                #     output_text = method["code"]
                #     out_f.write(json.dumps({"input": input_text, "output": output_text}, ensure_ascii=False) + "\n")