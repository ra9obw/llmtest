# model_name = "Qwen/Qwen2.5-7B"  # Убедись, что название модели корректное
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-1.7B"
# model_name = "Qwen/Qwen3-4B"
# model_name = "Qwen/Qwen3-8B"

# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import torch
# import time

# # Load model with 4-bit quantization
# model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# # Generation function with all modern parameters
# def generate_response(prompt, max_new_tokens=100):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=0.7,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,
#         use_cache=True,  # Explicitly enable cache
#     )
    
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Test the generation
# prompt = "Привет! Как тебя зовут?"
# t1 = time.time()
# response = generate_response(prompt)
# t2 = time.time()

# print(f"Response generated in {t2-t1:.2f} seconds")
# print("Response:", response)


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", 
                                             trust_remote_code=True, 
                                             quantization_config=quantization_config).cuda()
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))