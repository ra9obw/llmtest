from sentence_transformers import SentenceTransformer

# Скачиваем модель
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Проверяем путь к модели
print(model._model_path)



# from langchain.embeddings import HuggingFaceEmbeddings

# # Укажите путь к локальной модели
# local_model_path = "/path/to/your/local/model/all-MiniLM-L6-v2"

# # Инициализация эмбеддингов
# embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

# # Проверка работы модели
# text = "Это пример текста для проверки эмбеддингов."
# embedding = embeddings.embed_query(text)
# print(f"Размер эмбеддинга: {len(embedding)}")