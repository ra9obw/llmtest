from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Инициализация accelerator
accelerator = Accelerator()

# Шаг 1.1: Чтение файлов из кодовой базы
def read_codebase(directory, file = None):
    code_chunks = []
    if file == None:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and ("setup" not in file):  # Ограничимся Python-файлами
                    # print(type(file), file)
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        code_chunks.append(f.read())
    else:
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            code_chunks.append(f.read())
    return code_chunks

# Читаем кодовую базу
# code_chunks = read_codebase("C:\\work\\llm_test\\codebase\\adc4x250", "ADC4x250.cpp")
code_chunks = read_codebase("C:\\work\\llm_test\\codebase", "test.cpp")
print(f"Прочитано {len(code_chunks)} файлов.")

# Где хранится кэш моделей?
# C:\Users\<ВашеИмяПользователя>\.cache\huggingface\transformers\

model_name = "C:\\work\\llm_test\\models\\model-qwen3-14B-8b-tango-0"
# model_name = "Qwen/Qwen3-14B" # 40 layers with 30 is limit
bound = 30 
lr_count = 40
device_map = {
    "model.embed_tokens": "cuda:0",
    **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
    **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
    "model.norm": "cuda:0",
    "lm_head": "cuda:0"
}

# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-4B"
# model_name = "C:\\work\\llm_test\\models\\model-qwen3-4B-8b-tango-0"
# device_map = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Конфигурация для 4-bit квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Ускоряет вычисления на GPU
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    # device_map="auto"
    # device_map={"": "cuda:0"} 
    device_map=device_map
)

def ask_qestion(query, ans_szie = 10000):
    t1 = time.time()
    inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
    input_tokens_count = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=ans_szie,  # Уменьши до 50–100
        # max_new_tokens=50,  # Уменьши до 50–100
        do_sample=False,     # Отключи случайную генерацию
        num_beams=1          # Используй greedy decoding
    )
    t2 = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_tokens_count = outputs.shape[1]
    print(f"Ответ: {response}")
    print(f"Входной запрос содержит {input_tokens_count} токенов.")
    print(f"Сгенерированный ответ содержит {output_tokens_count} токенов.")
    print(f"Уникальных {output_tokens_count - input_tokens_count} токенов")
    print(f"Время генерации: {t2 - t1:.2f} секунд;\t\
          скорость: {output_tokens_count/(t2 - t1):.2f} ткн/с;\t\
          скорость относительно уникальных токенов: {(output_tokens_count - input_tokens_count)/(t2 - t1):.2f} ткн/с")

# Пример запроса
# prompt = "Привет! Как тебя зовут?"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
# query = [
#     # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Question: Как работает метод update в моей кодовой базе?",
#     # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Question: How the update method works in my codebase?",
#     # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Как работает метод update в классе RfChannelDs?",
#     "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "How the update method works in RfChannelDs?"
# ]

    # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "How does adc4x250 module interrupt handling work?"
query = [
     "Add doc-strings including @brief, @details @throws, @note, @param, @return or other necessary tags for С++ method in code below: " + code_chunks[0] +
     "do not add unnecessary tags, do not repeat the code, output just the doc string."
]

for idx, q in enumerate(query):
    ask_qestion(q)
    if idx != len(query)-1:
        txt = input("enter something to continue...")






# Освобождение памяти
del model
import gc
gc.collect()
torch.cuda.empty_cache()