from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Загрузка модели и токенизатора
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True
)

# Создание пайплайна для генерации текста
generate_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Обертка для LangChain
llm = HuggingFacePipeline(pipeline=generate_pipeline)

# Подготовка данных
texts = [
    "Qwen — это большая языковая модель от Alibaba Cloud.",
    "LangChain используется для создания приложений на основе LLM.",
    "RAG объединяет поиск и генерацию текста."
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(texts, embeddings)

# Настройка Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Создание RAG-цепочки
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Тестирование
query = "Что такое велосипед?"
response = qa_chain.run(query)
print(response)