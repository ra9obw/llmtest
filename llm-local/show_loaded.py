# from huggingface_hub import list_models

# # Вывести список всех доступных моделей (не только локальных)
# for model in list_models():
#     print(model.modelId)


# from huggingface_hub import list_models_in_cache

# # Получаем список всех моделей в кэше
# models = list_models_in_cache()

# print("Локально установленные модели:")
# for model in models:
#     print(f"- {model}")


import os

def get_installed_models():
    cache_dir = os.path.expanduser("C:\\Users\\fynjy\\.cache\\huggingface\\hub\\")
    installed_models = []

    for dir_name in os.listdir(cache_dir):
        if dir_name.startswith("models--"):
            parts = dir_name.split("--")
            if len(parts) == 3:
                owner = parts[1]
                model_id = parts[2]
                installed_models.append(f"{owner}/{model_id}")

    return installed_models

print("Установленные модели:")
for model in get_installed_models():
    print(f"- {model}")