import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Конфигурация
MODEL_NAME = "Qwen/Qwen3-14B"
LORA_ADAPTER_DIR ="C:\\work\\llm_test\\models\\model-qwen3-14B-8b-tango-0"
OUTPUT_DIR = "C:\\work\\llm_test\\models\\model-qwen3-14B-8b-tango-0_merged"

def merge_lora_with_base():
    # Загрузка исходной модели (ВАЖНО: без квантования!)
    print("Загружаем базовую модель...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Загрузка LoRA-адаптеров
    print("Загружаем LoRA-адаптеры...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    
    # Объединение и сохранение
    print("Объединяем модель с адаптерами...")
    merged_model = lora_model.merge_and_unload()
    
    print(f"Сохраняем объединённую модель в {OUTPUT_DIR}...")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Объединение завершено!")

if __name__ == "__main__":
    merge_lora_with_base()