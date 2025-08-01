import os
import gc
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch
# import bitsandbytes as bnb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use GPU 0

for i in range(1):
    torch.cuda.empty_cache()
    gc.collect()
    # Пути
    MODEL_NAME = "Qwen/Qwen2.5-7B"
    device_map = "cuda:0"
    OUTPUT_DIR = "C:\\work\\llm_test\\models\\model-qwen2-7B-8b-tango-0"
 
    DATASET_PATH = "C:\\work\\llm_test\\dataset_lora.jsonl"

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    # Квантование 4-bit + LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    config = LoraConfig(
        r=32, #32  # ранг LoRA матрицы
        lora_alpha=64, #64
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Загрузка модели с квантованием 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        # device_map="cuda:0",
        # device_map = "auto",
        device_map = device_map,
        trust_remote_code=True
    )

    # Применяем LoRA
    model = get_peft_model(model, config)

    def tokenize_function(examples):
        # Создаем списки для хранения информации об обрезании
        input_truncated = []
        output_truncated = []
        # Токенизируем все примеры
        inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
        outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
        # Проверяем каждый пример на обрезание
        for i in range(len(examples["input"])):
            # Проверяем вход
            input_tokens_no_trunc = tokenizer(examples["input"][i], truncation=False)
            input_truncated.append(len(input_tokens_no_trunc["input_ids"]) > 512)
            # Проверяем выход
            output_tokens_no_trunc = tokenizer(examples["output"][i], truncation=False)
            output_truncated.append(len(output_tokens_no_trunc["input_ids"]) > 512)
        inputs['labels'] = outputs["input_ids"]
        inputs['input_truncated'] = input_truncated
        inputs['output_truncated'] = output_truncated
        return inputs

    # Загрузка датасета
    dataset = load_dataset("json", data_files=DATASET_PATH)
    # Разделение на train и test (например, 90% train, 10% test)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

    tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,  # Добавляем batch size для оценки
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,  # Сохраняем только последние 2 чекпоинта
        learning_rate=2e-5,
        gradient_accumulation_steps=8,
        report_to="tensorboard",
        fp16=True,
        eval_strategy="no",  # Добавляем оценку
        # eval_steps=1000,  # Оцениваем каждые 500 шагов
    )

    # Коллекция данных
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Создаём trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator
    )

    # Запуск обучения
    print("Начинаем дообучение...")
    trainer.train()

    # Сохранение модели
    print("Сохраняем модель...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Оценка модели
    print("Оценка модели...")
    eval_results = trainer.evaluate()
    print(f"Loss: {eval_results['eval_loss']}")