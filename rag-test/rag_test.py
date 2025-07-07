from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import time

FIASS_INDEX = "faiss_index_ds"

# Загрузка индекса
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    FIASS_INDEX, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Загрузка локальной LLM
# model_name = "Qwen/Qwen2.5-7B"  # Например, путь к модели GPT-J или Llama
# model_name = "D:\\work\\llm_test312\\fine-tuning\\model"
# model_name = "Qwen/Qwen-Coder-14B"
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "D:\\work\\llm_test312\\fine-tuning\\model-qwen3-0.6B-32b"

# model_name = "Qwen/Qwen3-14B" # 40 layers with 30 is limit
# bound = 30 
# lr_count = 40
# device_map = {
#     "model.embed_tokens": "cuda:0",
#     **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
#     **{f"model.layers.{i}": "cuda:1" for i in range(bound, middle_bound)},
#     **{f"model.layers.{i}": "cpu" for i in range(middle_bound, lr_count)},
#     "model.norm": "cuda:0",
#     "lm_head": "cuda:0"
# }

# model_name = "Qwen/Qwen2.5-32B"
model_name = "Qwen/Qwen3-30B-A3B"
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
# Конфигурация для 4-bit квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Ускоряет вычисления на GPU
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, #use for "Qwen/Qwen3-30B-A3B" multi device map
    torch_dtype=torch.float16,  # Используй float16 для GPU
    # load_in_4bit=True,          # Активируй 4-bit квантизацию
    # device_map="auto"           # Автоматическое распределение по устройствам
    # device_map={"": "cuda:0"}
    device_map = device_map
)

# Создание pipeline для генерации текста
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device_map="auto"  # Убедитесь, что pipeline также использует device_map
    device_map={"": "cuda:0"}
)

# Создание цепочки RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=llm_pipeline),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# Пример запроса

query = [
    # "Как работает метод update в классе RfChannelDs?",
    # "Как работает метод update в моей кодовой базе?",
    # "How the update method works in my codebase?",
    "How the update method works in RfChannelDs? "
]

for q in query:
    t1 = time.time()
    response = qa_chain.run(q)
    t2 = time.time()
    print(response)
    print("Time lapsed: ", t2-t1)
    input("Enter to continue...")