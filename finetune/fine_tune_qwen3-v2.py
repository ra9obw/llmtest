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
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq 
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch
# import bitsandbytes as bnb
device_map = "auto"
# device_map = "cuda:0"
MODEL_NAME = "Qwen/Qwen3-4B"  # или локальный путь
OUTPUT_DIR_FORMAT = "C:\\work\\llm_test\\models\\model-qwen3-4B-8b-tangodoc-{i}"

# MODEL_NAME = "Qwen/Qwen3-4B"  # или локальный путь
# OUTPUT_DIR = f"C:\\work\\llm_test\\models\\model-qwen3-4B-8b-tango-{i}"

# MODEL_NAME = "Qwen/Qwen3-32B"
# OUTPUT_DIR = f"C:\\work\\llm_test\\models\\model-qwen3-32B-8b-tango-{i}"
# bound = 35 
# lr_count = 64
# device_map = {
#     "model.embed_tokens": "cuda:0",
#     **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
#     **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
#     "model.norm": "cuda:0",
#     "lm_head": "cuda:0"
# }

# MODEL_NAME = "Qwen/Qwen2.5-7B"
# device_map = "cuda:0"
# OUTPUT_DIR = "C:\\work\\llm_test\\models\\model-qwen2-7B-8b-tango-0"
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
DATASET_PATH = "C:\\work\\llm_test\\dataset_doc_finetune_cppTango-9.3.7.jsonl"



os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only use GPU 1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
for idx in range(1):
    torch.cuda.empty_cache()
    gc.collect()
    # Пути
    OUTPUT_DIR = OUTPUT_DIR_FORMAT.format(i=idx)
    
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем eos_token как pad_token
    
    # Квантование 4-bit + LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Загрузка модели с квантованием 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

    # Применяем LoRA
    model = get_peft_model(model, config)

    def tokenize_function(examples):
        # Токенизируем вход и выход без обрезки и padding
        inputs = tokenizer(examples["input"], truncation=False, padding=False)
        outputs = tokenizer(examples["output"], truncation=False, padding=False)
        
        # Создаем полные последовательности для обучения
        full_sequences = []
        for input_ids, output_ids in zip(inputs["input_ids"], outputs["input_ids"]):
            # Объединяем input и output с разделителем (если нужно)
            sequence = input_ids + output_ids + [tokenizer.eos_token_id]
            full_sequences.append(sequence)
            
        return {"input_ids": full_sequences}

    # Загрузка датасета
    dataset = load_dataset("json", data_files=DATASET_PATH)
    # Разделение на train и test
    split_dataset = dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

    # Токенизация без padding
    tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

    # Специальный data collator для динамического padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,  # Опционально: выравнивание для эффективности GPU
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-5,
        gradient_accumulation_steps=8,
        report_to="tensorboard",
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
        # Оптимизации для больших последовательностей
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,  # Группировка по длине для эффективности
    )

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