import json
from pathlib import Path

def convert_to_lora_format(input_file: str, output_file: str) -> None:
    """
    Convert the extracted code structures JSONL file to LoRA training format.
    
    Args:
        input_file: Path to input JSONL file with extracted code structures
        output_file: Path to output JSONL file in LoRA format
    """
    type_to_prompt = {
        "class": "Implement class with name {name}",
        "method": "Implement method with name {name}",
        "class_method": "Implement method of class {class_name} with name {name}",
        "function": "Implement function with name {name}",
        "template": "Implement template with name {name}",
        "class_template": "Implement class template with name {name}",
        "namespace": "Implement namespace with name {name}",
        "lambda": "Implement lambda expression",
        "error_handler": "Implement error handler",
        "macro": "Implement macro with name {name}",
        "preprocessor": "Implement preprocessor directive",
        "literal": "Implement user-defined literal with name {name}",
        "attribute": "Implement attribute with name {name}"
    }
    stop_list = ["namespace", "lambda", "error_handler"]
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                item = json.loads(line.strip())
                item_type = item["type"]
                
                # Get the appropriate prompt template
                prompt_template = type_to_prompt.get(item_type, "Implement {type} with name {name}")
                if item_type in stop_list:
                    continue
                # Prepare input text
                if item_type == "class":
                    # For classes, we might want to include methods in the output
                    input_text = prompt_template.format(name=item["name"])
                    output_code = item.get("declaration", "")
                    lora_item = {
                        "input": input_text,
                        "output": output_code.strip()
                    }
                    json.dump(lora_item, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                    if "methods" in item and item["methods"]:
                        for m in item["methods"]:
                            prmt =  type_to_prompt["class_method"]
                            input_text = prmt.format(class_name=item["name"], name = m["name"])
                            output_code = m.get("code", "")
                            lora_item = {
                                "input": input_text,
                                "output": output_code.strip()
                            }
                            
                            # Write to output file
                            json.dump(lora_item, outfile, ensure_ascii=False)
                            outfile.write('\n')
                else:
                    input_text = prompt_template.format(name=item.get("name", ""), type=item_type)
                    output_code = item.get("code", "")
                
                    # Create LoRA training example
                    lora_item = {
                        "input": input_text,
                        "output": output_code.strip()
                    }
                    
                    # Write to output file
                    json.dump(lora_item, outfile, ensure_ascii=False)
                    outfile.write('\n')
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except KeyError as e:
                print(f"Missing key in item: {e}")

if __name__ == "__main__":
    # Configuration
    INPUT_JSONL = r"E:\\work\\llm_test\\dataset_clang.jsonl"
    OUTPUT_JSONL = r"E:\\work\\llm_test\\dataset_lora.jsonl"
    
    # Convert the dataset
    print(f"Converting {INPUT_JSONL} to LoRA format...")
    convert_to_lora_format(INPUT_JSONL, OUTPUT_JSONL)
    print(f"LoRA dataset saved to {OUTPUT_JSONL}")