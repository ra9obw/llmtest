from transformers import AutoModelForCausalLM
import re

# Загрузка только для получения информации о структуре
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", 
                                             local_files_only=True)

layer_numbers = set()
for name, _ in model.named_parameters():
    match = re.search(r'model\.layers\.(\d+)', name)
    if match:
        layer_numbers.add(int(match.group(1)))

print(f"Количество слоёв: {len(layer_numbers)}")