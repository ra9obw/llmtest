from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Шаг 1.1: Чтение файлов из кодовой базы
def read_codebase(directory):
    code_chunks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and ("setup" not in file):  # Ограничимся Python-файлами
                # print(type(file), file)
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code_chunks.append(f.read())
    return code_chunks

# Читаем кодовую базу
code_chunks = read_codebase("D:\\work\\llm_test312\\rag-test\\my_codebase")
print(f"Прочитано {len(code_chunks)} файлов.")

model_name = "Qwen/Qwen3-30B-A3B"
# bound = 23 
# lr_count = 48

# model_name = "Qwen/Qwen3-14B" # 40 layers with 30 is limit
# bound = 30 
# lr_count = 40

# model_name = "Qwen/Qwen2.5-7B"  # Убедись, что название модели корректное
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Конфигурация для 4-bit квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Ускоряет вычисления на GPU
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)
# Где хранится кэш моделей?
# C:\Users\<ВашеИмяПользователя>\.cache\huggingface\transformers\
# Загрузка модели с 4-bit квантизацией
# Загрузка модели с 4-bit квантизацией
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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Используем новую систему конфигурации
    torch_dtype=torch.float16,
    # device_map="auto"
    # device_map={"": "cuda:0"}  # Можно заменить на "auto"
    device_map = device_map
)
# for name, param in model.named_parameters():
#     # print(name)
#     # if "transformer.h" in name:
#     if True:
#         # Объём памяти в байтах: количество элементов × размер одного элемента
#         size_in_bytes = param.numel() * param.element_size()
#         size_in_mb = size_in_bytes / (1024 ** 2)  # Конвертация в Мб
#         print(f"{name}: {size_in_mb:.2f} MB")

# Пример запроса
# query = "Привет! Как тебя зовут?"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
query = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Question: Как работает метод update в моей кодовой базе?"

t1 = time.time()
inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
input_tokens_count = inputs["input_ids"].shape[1]

outputs = model.generate(
    **inputs,
    max_new_tokens=600,  # Уменьши до 50–100
    # max_new_tokens=50,  # Уменьши до 50–100
    do_sample=False,     # Отключи случайную генерацию
    num_beams=1          # Используй greedy decoding
)
output_tokens_count = outputs.shape[1]

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
t2 = time.time()

print(f"Входной запрос содержит {input_tokens_count} токенов.")
print(f"Сгенерированный ответ содержит {output_tokens_count} токенов.")
print(f"Время генерации: {t2 - t1:.2f} секунд, {(output_tokens_count/(t2 - t1)):.2f} tokens/s")
print(response)


# Освобождение памяти
del model
import gc
gc.collect()
torch.cuda.empty_cache()