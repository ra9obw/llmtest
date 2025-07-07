from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Шаг 1.1: Чтение файлов из кодовой базы
def read_codebase(directory):
    code_chunks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # Ограничимся Python-файлами
                # print(type(file))
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code_chunks.append(f.read())
    return code_chunks

# Шаг 1.2: Разделение кода на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Читаем кодовую базу
code_chunks = read_codebase("D:\\work\\llm_test312\\rag-test\\my_codebase")
print(f"Прочитано {len(code_chunks)} файлов.")

# Разделяем текст на чанки (используем split_text вместо split_documents)
split_texts = []
for chunk in code_chunks:
    split_texts.extend(text_splitter.split_text(chunk))

print(f"Разделено на {len(split_texts)} чанков.")

# Шаг 1.3: Создание эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Шаг 1.4: Индексация в FAISS
# Создаем объекты Document для FAISS
from langchain.schema import Document
documents = [Document(page_content=text) for text in split_texts]

vector_store = FAISS.from_documents(documents, embeddings)

# Сохранение индекса
vector_store.save_local("faiss_index")