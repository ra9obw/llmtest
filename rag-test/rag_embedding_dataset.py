from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

# Шаг 1: Загрузка данных
with open("D:\\work\\llm_test312\\rag-test\\dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Шаг 2: Предобработка текста
def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.startswith('#')]
    return '\n'.join(cleaned_lines).strip()

split_texts = []
for item in data:
    content = clean_text(item["output"])
    split_texts.append(content)

# Шаг 3: Создание эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Шаг 4: Создание документов
from langchain.schema import Document
documents = [Document(page_content=text) for text in split_texts]

# Шаг 5: Индексация в FAISS
vector_store = FAISS.from_documents(documents, embeddings)

# Шаг 6: Сохранение индекса
vector_store.save_local("faiss_index_ds")