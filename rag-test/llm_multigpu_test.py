import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen3-30B-A3B"

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Конфигурация для 4-bit квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# device_map с частичным размещением на CPU
bound = 15
middle_bound = 25
lr_count = 48

device_map = {
    "model.embed_tokens": "cuda:0",
    **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
    **{f"model.layers.{i}": "cuda:1" for i in range(bound, middle_bound)},
    **{f"model.layers.{i}": "cpu" for i in range(middle_bound, lr_count)},
    "model.norm": "cuda:0",
    "lm_head": "cuda:0"
}

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map=device_map
)

# Входной запрос
prompt = "Привет! Как тебя зовут?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Генерация текста
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=False,
    num_beams=1
)

# Декодирование и вывод
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Освобождение памяти
del model
import gc
gc.collect()
torch.cuda.empty_cache()