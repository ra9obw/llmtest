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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only use GPU 1
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
for i in range(1):
    torch.cuda.empty_cache()
    gc.collect()
    # Пути
    # MODEL_NAME = "Qwen/Qwen3-8B"  # или локальный путь
    # OUTPUT_DIR = f"C:\\work\\llm_test\\models\\model-qwen3-8B-8b-tango-{i}"

    MODEL_NAME = "Qwen/Qwen3-4B"  # или локальный путь
    # OUTPUT_DIR = f"C:\\work\\llm_test\\models\\model-qwen3-4B-8b-tango-{i}"
    OUTPUT_DIR = f"C:\\work\\llm_test\\models\\model-qwen3-4B-8b-tango-test"
    device_map = "auto"
    DATASET_PATH = "C:\\work\\llm_test\\dataset_code_finetune_cppTango-9.3.7.jsonl"

    # MODEL_NAME = "Qwen/Qwen3-32B"
    # OUTPUT_DIR = f"C:\\work\\llm_test\\models\\model-qwen3-32B-8b-tango-{i}"
    # bound = 36 
    # lr_count = 64
    # device_map = {
    #     "model.embed_tokens": "cuda:0",
    #     **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
    #     **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
    #     "model.norm": "cuda:0",
    #     "lm_head": "cuda:0"
    # }

    # MODEL_NAME = "Qwen/Qwen2.5-7B"
    # OUTPUT_DIR = "D:\\work\\llm_test312\\fine-tuning\\model"

    # MODEL_NAME = "Qwen/Qwen3-0.6B"
    # OUTPUT_DIR = "D:\\work\\llm_test312\\fine-tuning\\model-qwen3-0.6B-32b"

    # MODEL_NAME = "Qwen/Qwen3-14B"
    # bound = 24 
    # lr_count = 40
    # device_map = {
    #     "model.embed_tokens": "cuda:0",
    #     **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
    #     **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
    #     "model.norm": "cuda:0",
    #     "lm_head": "cuda:0"
    # }
    # OUTPUT_DIR = "C:\\work\\llm_test\\models\\model-qwen3-14B-8b-tango-0"

    # DATASET_PATH = "D:\\work\\llm_test312\\fine-tuning\\dataset.json"
    # DATASET_PATH = "C:\\work\\llm_test\\dataset_lora.jsonl"


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
        r=8, #32  # ранг LoRA матрицы
        lora_alpha=16, #64
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

    # model = torch.compile(model)

    # Применяем LoRA
    model = get_peft_model(model, config)

    # Функция токенизации
    # def tokenize_function(examples):
    #     inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    #     outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
    #     inputs['labels'] = outputs["input_ids"]
    #     return inputs

    def tokenize_function(examples):
        # Создаем списки для хранения информации об обрезании
        # input_truncated = []
        # output_truncated = []
        # Токенизируем все примеры
        inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=2048)
        outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=2048)
        # # Проверяем каждый пример на обрезание
        # for i in range(len(examples["input"])):
        #     # Проверяем вход
        #     input_tokens_no_trunc = tokenizer(examples["input"][i], truncation=False)
        #     input_truncated.append(len(input_tokens_no_trunc["input_ids"]) > 512)
        #     # Проверяем выход
        #     output_tokens_no_trunc = tokenizer(examples["output"][i], truncation=False)
        #     output_truncated.append(len(output_tokens_no_trunc["input_ids"]) > 512)
        inputs['labels'] = outputs["input_ids"]
        # inputs['input_truncated'] = input_truncated
        # inputs['output_truncated'] = output_truncated
        return inputs

    # def tokenize_function(examples):
    #     # Tokenize inputs and outputs separately
    #     model_inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
        
    #     # Tokenize outputs with the same tokenizer
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
        
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs

    # Загрузка датасета
    dataset = load_dataset("json", data_files=DATASET_PATH)
    # Разделение на train и test (например, 90% train, 10% test)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

    tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

    # Настройки тренировки
    # training_args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     per_device_train_batch_size=1,
    #     per_device_eval_batch_size=1,  # Добавляем batch size для оценки
    #     num_train_epochs=3,
    #     logging_dir="./logs",
    #     logging_steps=100,
    #     save_steps=500,
    #     save_total_limit=2,  # Сохраняем только последние 2 чекпоинта
    #     learning_rate=2e-5,
    #     gradient_accumulation_steps=8,
    #     report_to="tensorboard",
    #     fp16=True,
    #     eval_strategy="steps",  # Добавляем оценку
    #     eval_steps=1000,  # Оцениваем каждые 500 шагов
    # )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,  # Добавляем batch size для оценки
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,  # Сохраняем только последние 2 чекпоинта
        learning_rate=2e-5,
        gradient_accumulation_steps=8,
        # report_to="tensorboard",
        optim="adafactor",
        fp16=True,
        eval_strategy="no",  # Добавляем оценку
        # eval_steps=1000,  # Оцениваем каждые 500 шагов
    )

    # training_args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     per_device_train_batch_size=1,
    #     gradient_accumulation_steps=8,
    #     num_train_epochs=3,
    #     learning_rate=2e-5,
    #     logging_dir="./logs",
    #     logging_steps=100,
    #     save_steps=500,
    #     save_total_limit=2,
    #     fp16=True,
    #     eval_strategy="no",
    #     gradient_checkpointing=True,
    #     optim="adafactor",
    #     # Add these to prevent the warnings
    #     remove_unused_columns=False,  # Important for keeping labels
    #     label_names=["input_ids", "labels"]  # Explicitly specify label names
    # )

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