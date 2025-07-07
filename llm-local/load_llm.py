from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = "Qwen/Qwen2.5-7B"  # Убедись, что название модели корректное
# model_name = "Qwen/Qwen3-7B"
# model_name = "Qwen/Qwen3-14B"
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-1.7B"
# model_name = "Qwen/Qwen3-4B"
model_name = "Qwen/Qwen3-8B"


tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка модели с 4-bit квантизацией
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Используй float16 для GPU
    load_in_4bit=True,          # Активируй 4-bit квантизацию
    device_map="auto"           # Автоматическое распределение по устройствам
)