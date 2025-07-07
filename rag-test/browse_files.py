import os

# Шаг 1.1: Чтение файлов из кодовой базы
def read_codebase(directory):
    code_chunks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and ("setup" not in file):  # Ограничимся Python-файлами
                # print(type(file), file)
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code_chunks.append(f.read())
    return code_chunks

# Читаем кодовую базу
code_chunks = read_codebase("D:\\work\\llm_test312\\rag-test\\my_codebase")
print(f"Прочитано {len(code_chunks)} файлов.")