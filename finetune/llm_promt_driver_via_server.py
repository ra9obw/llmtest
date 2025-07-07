import requests
import json
import time
import os
from typing import Optional, Dict, List


class LLMPromptDriver:
    def __init__(self, model_name: Optional[str] = None, api_base: str = "http://localhost:1234/v1"):
        """
        :param model_name: Имя модели (игнорируется LM Studio, но можно указать для совместимости)
        :param api_base: URL сервера LM Studio (по умолчанию http://localhost:1234/v1)
        """
        self.api_base = api_base
        self.model_name = model_name or "local-model"  # LM Studio игнорирует это поле
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer lm-studio"  # Фиктивный ключ (LM Studio не требует аутентификации)
        }

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Нет необходимости в очистке для HTTP-клиента
    
    def initialize(self):
        pass

    def finalize(self):
        pass

    def ask_question(self, query: str, ans_size: int = 1024, **kwargs) -> str:
        """
        Отправляет запрос к LM Studio API.
        
        :param query: Текст запроса
        :param ans_size: Максимальное количество токенов в ответе
        :param kwargs: Дополнительные параметры (temperature, top_p и т.д.)
        :return: Ответ модели
        """
        messages = [
            {"role": "system", "content": "You are an experienced software developer"},
            {"role": "user", "content": query}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": ans_size,
            **kwargs  # Дополнительные параметры (temperature, top_p и т.д.)
        }

        try:
            t1 = time.time()
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()  # Проверка на ошибки HTTP
            
            data = response.json()
            t2 = time.time()
            
            # Логирование статистики
            if "usage" in data:
                usage = data["usage"]
                print(f"Токенов: вход {usage['prompt_tokens']}, выход {usage['completion_tokens']}")
            
            answer = data["choices"][0]["message"]["content"]
            print(f"Время генерации: {t2 - t1:.2f} сек; скорость {(usage['completion_tokens'])/(t2 - t1):.2f}")
            return answer.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к LM Studio: {e}")
            raise
        except json.JSONDecodeError:
            print(f"Ошибка разбора JSON: {response.text}")
            raise

    def ask_batched_questions(self, queries: List[str], ans_size: int = 10000, **kwargs) -> List[str]:
        """
        Sends multiple queries to LM Studio API sequentially (since batch processing isn't natively supported).
        
        :param queries: List of text queries
        :param ans_size: Maximum number of tokens in each response
        :param kwargs: Additional parameters (temperature, top_p etc.)
        :return: List of model responses in the same order as queries
        """
        answers = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_time = 0
        
        for query in queries:
            try:
                t1 = time.time()
                messages = [
                    {"role": "system", "content": "You are an experienced software developer"},
                    {"role": "user", "content": query}
                ]
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": ans_size,
                    **kwargs
                }
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                
                data = response.json()
                t2 = time.time()
                
                if "usage" in data:
                    usage = data["usage"]
                    total_prompt_tokens += usage['prompt_tokens']
                    total_completion_tokens += usage['completion_tokens']
                
                answers.append(data["choices"][0]["message"]["content"].strip())
                total_time += (t2 - t1)
                
            except Exception as e:
                print(f"Error processing query: {query[:50]}...: {e}")
                answers.append("")  # Or raise the exception if you prefer
        
        # Print aggregated statistics
        if total_prompt_tokens > 0:
            print(f"Total tokens: prompt {total_prompt_tokens}, completion {total_completion_tokens}")
            print(f"Total time: {total_time:.2f} sec; "
                f"Average speed: {total_completion_tokens/total_time:.2f} tokens/sec")
        
        return answers

    def stream_question(self, query: str, ans_size: int = 1024, **kwargs):
        """
        Потоковый вывод ответа (если LM Studio поддерживает stream=True).
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": ans_size,
            "stream": True,
            **kwargs
        }

        try:
            with requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True
            ) as response:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        if "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
        except Exception as e:
            print(f"Ошибка потокового запроса: {e}")
            raise



# Шаг 1.1: Чтение файлов из кодовой базы
def read_codebase(directory, file = None):
    code_chunks = []
    if file == None:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and ("setup" not in file):  # Ограничимся Python-файлами
                    # print(type(file), file)
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        code_chunks.append(f.read())
    else:
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            code_chunks.append(f.read())
    return code_chunks

def ask_from_file():
    input_file = "C:\\work\\llm_test\\finetune\\promt.txt"
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
        print("File content:")
        print(content, type(content))
        return content

if __name__ == "__main__":

    # Читаем кодовую базу
    # code_chunks = read_codebase("C:\\work\\llm_test\\codebase\\adc4x250", "ADC4x250.cpp")
    # code_chunks = read_codebase("C:\\work\\llm_test\\codebase", "test.cpp")
    # print(f"Прочитано {len(code_chunks)} файлов.")


    # asking = [
    #     "Add doc-strings in English before the C++ method.",
    #     "Use tags: @brief, @param, @return, @throws, @note",
    #     "Code:",
    #     code_chunks[0],
    #     "Use this template:",
    #     "/**",
    #     " * @brief [one-line description]",
    #     " * @param [name] [description]",
    #     " * @return [description]",
    #     " * @throws [exception] [reason]",
    #     " * @note [optional note]",
    #     " */",
    #     "do not repeat the code, output just docstring"
    # ]
    # query = ["\n".join(asking)]

    # query = [ask_from_file()]

    # for idx, q in enumerate(query):
    #     with LLMPromptDriver() as llm:
    #         answer = llm.ask_question(q, temperature=0.2)
    #         print(answer)

    with LLMPromptDriver() as llm:
        queries = [
        "Расскажи о квантовой физике",
        "Напиши стихотворение про ИИ и матрицу",
        "Как работает GPT-4?"
        ]
        ans = llm.ask_batched_questions(queries, 100)
        print(ans)
