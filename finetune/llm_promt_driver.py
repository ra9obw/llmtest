from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch
import os
import time
import gc
import re
from typing import Optional, Dict, List


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Инициализация accelerator
accelerator = Accelerator()

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

# Где хранится кэш моделей?
# C:\Users\<ВашеИмяПользователя>\.cache\huggingface\transformers\

class llm_promt_driver:
    def __init__(self, model_name = None):
        if model_name == None:
            # self.model_name = "Qwen/Qwen3-8B" # 40 layers with 30 is limit
            # bound = 24 
            # lr_count = 36
            # self.device_map = {
            #     "model.embed_tokens": "cuda:0",
            #     **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
            #     **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
            #     "model.norm": "cuda:0",
            #     "lm_head": "cuda:0"
            # }
            # self.model_name = "C:\\work\\llm_test\\models\\model-qwen3-14B-8b-tango-0"
            self.model_name = "Qwen/Qwen3-14B" # 40 layers with 30 is limit
            bound = 30 
            lr_count = 40
            self.device_map = {
                "model.embed_tokens": "cuda:0",
                **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
                **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
                "model.norm": "cuda:0",
                "lm_head": "cuda:0"
            }

            # self.model_name = "Qwen/Qwen3-0.6B"
            # self.model_name = "Qwen/Qwen3-4B"
            # self.model_name = "C:\\work\\llm_test\\models\\model-qwen3-4B-8b-tango-0"
            # self.device_map = "auto"
            # self.device_map = "cuda:0"

            # bound = 38
            # # middle_bound = bound + 10
            # middle_bound = bound + 18
            # lr_count = 64
            # self.model_name = "Qwen/Qwen3-32B"
            # self.device_map = {
            #     "model.embed_tokens": "cuda:0",
            #     **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
            #     **{f"model.layers.{i}": "cuda:1" for i in range(bound, middle_bound)},
            #     **{f"model.layers.{i}": "cpu" for i in range(middle_bound, lr_count)},
            #     "model.norm": "cuda:0",
            #     "lm_head": "cuda:0"
            # }
        else:
            self.model_name = model_name
            if model_name in ("Qwen/Qwen3-14B", "C:\\work\\llm_test\\models\\model-qwen3-14B-8b-tango-0"):
                bound = 20 
                lr_count = 40
                self.device_map = {
                "model.embed_tokens": "cuda:0",
                **{f"model.layers.{i}": "cuda:0" for i in range(0, bound)},
                **{f"model.layers.{i}": "cuda:1" for i in range(bound, lr_count)},
                "model.norm": "cuda:0",
                "lm_head": "cuda:0"
                }
            else:
                self.device_map = "cuda:0"

    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def initialize(self):
        t1 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Конфигурация для 4-bit квантизации
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Ускоряет вычисления на GPU
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,  # Основное изменение - используем 8-bit вместо 4-bit
        #     llm_int8_enable_fp32_cpu_offload=True
        # )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            # torch_dtype=torch.bfloat16,
            torch_dtype=torch.float16,
            device_map=self.device_map
        )
        t2 = time.time()
        print(f"Время загрузки модели: {(t2 - t1):.2f}")

    def tokenize_input(self, query):
        t1 = time.time()
        print(f"Входной запрос содержит {query}")
        inputs = self.tokenizer(query, return_tensors="pt", padding=True).to("cuda:0")
        input_tokens_count = inputs["input_ids"].shape[1]
        t2 = time.time()
        if t2 - t1 < 1.0e-3:
            print(f"Входной запрос содержит {input_tokens_count} токенов. Затраченное время: {(t2 - t1):.2f}")
        else:
            print(f"Входной запрос содержит {input_tokens_count} токенов. Скорость токенизации {input_tokens_count/(t2 - t1):.2f} ткн/с")
        print(f"len(inputs):\t{len(inputs)}")
        return inputs, input_tokens_count

    def ask_tokenized_question(self, inputs, input_tokens_count, ans_size = 10000):
        try:
            t1 = time.time()
            outputs = self.model.generate(
                **inputs,
                temperature=0.2,
                max_new_tokens=ans_size,  # Уменьши до 50–100
                do_sample=False,     # Отключи случайную генерацию
                num_beams=5,          # Используй greedy decoding
                repetition_penalty=1.5,  # Штраф за повторения
                early_stopping=True  # Остановка при естественном завершении
            )
            t2 = time.time()
            response = self.tokenizer.decode(outputs[0][input_tokens_count:], skip_special_tokens=True)
            print(f"len(outputs):\t{len(outputs)}")
            output_tokens_count = outputs.shape[1]
            print(f"Сгенерирован ответ {response}")
            print(f"Сгенерированный ответ содержит {output_tokens_count} токенов.")
            print(f"Уникальных {output_tokens_count - input_tokens_count} токенов")
            print(f"Время генерации: {t2 - t1:.2f} секунд;\t\
                скорость: {output_tokens_count/(t2 - t1):.2f} ткн/с;\t\
                скорость относительно уникальных токенов: {(output_tokens_count - input_tokens_count)/(t2 - t1):.2f} ткн/с")
            response = response.strip()
            del inputs
            if 'outputs' in locals():
                del outputs
            torch.cuda.synchronize()  # Ожидаем завершения операций CUDA
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            torch.cuda.empty_cache()
            raise e        
        # Дополнительно ищем \think (если нужен)
        think_tag = "</think>"
        if think_tag in response:
            parts = response.split(think_tag, 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return response    

    def ask_batched_questions(self, queries: List[str], ans_size: int = 10000) -> List[str]:
        """
        Обрабатывает несколько текстовых запросов в батче
        
        :param queries: Список текстовых запросов
        :param ans_size: Максимальное количество новых токенов в ответе
        :return: Список ответов
        """
        try:
            # Токенизация всех запросов с паддингом
            inputs = self.tokenizer(queries, padding=True, return_tensors="pt").to("cuda:0")
            input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
            
            t1 = time.time()
            
            # Генерация ответов для всего батча
            outputs = self.model.generate(
                **inputs,
                temperature=0.2,
                max_new_tokens=ans_size,
                do_sample=False,
                num_beams=5,
                repetition_penalty=1.5,
                early_stopping=True
            )
            
            t2 = time.time()
            
            # Декодирование каждого ответа
            responses = []
            total_tokens = 0
            total_unique = 0
            
            for i in range(len(queries)):
                # Вырезаем только сгенерированную часть
                output_seq = outputs[i][input_lengths[i]:]
                response = self.tokenizer.decode(output_seq, skip_special_tokens=True).strip()
                
                # Обработка тега </think>
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()
                
                responses.append(response)
                
                # Статистика
                output_len = len(output_seq)
                total_tokens += len(outputs[i])
                total_unique += output_len
                
                print(f"\nОтвет {i+1}/{len(queries)}:")
                print(f"Уникальных токенов: {output_len}")
                print(f"Ответ: {response[:200]}...")

            # Общая статистика
            batch_time = t2 - t1
            print(f"\nОбщая статистика батча:")
            print(f"Запросов: {len(queries)}")
            print(f"Время обработки: {batch_time:.2f} сек")
            print(f"Скорость генерации: {total_unique/batch_time:.2f} токенов/сек")
            print(f"Всего токенов: {total_tokens}")
            
            return responses
            
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def ask_question(self, query, ans_size = 1024):
        inputs, input_tokens_count = self.tokenize_input(query)
        ans_tkn_count = min(ans_size, max(2048, 2*input_tokens_count))
        response = self.ask_tokenized_question(inputs, input_tokens_count, ans_tkn_count)
        return response

    def finalize(self):
        # Освобождение памяти
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def clean_response(response):
    # Удаляем всё до первого тега
    first_tag = find_first_tag(response)
    if first_tag:
        response = response[response.find(first_tag):]
    
    # Удаляем всё после последнего тега
    last_tag = find_last_tag(response)
    if last_tag:
        end_pos = response.rfind(last_tag) + len(last_tag)
        response = response[:end_pos]
    
    # Удаляем повторяющиеся строки
    lines = []
    seen = set()
    for line in response.split('\n'):
        clean_line = line.strip()
        if clean_line and clean_line not in seen:
            seen.add(clean_line)
            lines.append(line)
    
    return '\n'.join(lines)

def find_first_tag(text):
    tags = ['@brief', '@param', '@return', '@throws', '@note', '@details']
    for tag in tags:
        if tag in text:
            return tag
    return None

def find_last_tag(text):
    tags = ['@brief', '@param', '@return', '@throws', '@note', '@details']
    last_pos = -1
    result = None
    for tag in tags:
        pos = text.rfind(tag)
        if pos > last_pos:
            last_pos = pos
            result = tag
    return result

def extract_docstrings(text):
    """
    Извлекает все докстринги из переданного текста.
    
    Аргументы:
        text (str): Исходный текст, который может содержать докстринги.
        
    Возвращает:
        str: Текст, содержащий только докстринги, или пустую строку, если докстрингов нет.
    """
    # Регулярное выражение для поиска многострочных докстрингов (Python и C/C++ стиль)
    pattern = r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|/\*\*[\s\S]*?\*/)'
    docstrings = re.findall(pattern, text)
    return '\n'.join(docstrings) if docstrings else ''

if __name__ == "__main__":

    # Читаем кодовую базу
    # code_chunks = read_codebase("C:\\work\\llm_test\\codebase\\adc4x250", "ADC4x250.cpp")
    # code_chunks = read_codebase("C:\\work\\llm_test\\codebase", "test.cpp")
    # print(f"Прочитано {len(code_chunks)} файлов.")


    llm = llm_promt_driver()
    llm.initialize()

    queries = [
    "Расскажи о квантовой физике",
    "Напиши стихотворение про ИИ и матрицу",
    "Как работает GPT-4?"
    ]
    ans = llm.ask_batched_questions(queries, 100)
    # Пример запроса
    # prompt = "Привет! Как тебя зовут?"
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    # query = [
    #     # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Question: Как работает метод update в моей кодовой базе?",
    #     # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Question: How the update method works in my codebase?",
    #     # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "Как работает метод update в классе RfChannelDs?",
    #     "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "How the update method works in RfChannelDs?"
    # ]

        # "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer." + code_chunks[0] + "How does adc4x250 module interrupt handling work?"
    # asking = [
    #     "Add doc-strings including @brief, @details @throws, @note, @param, @return or other necessary tags for С++ method in code below: ",
    #     code_chunks[0],
    #     "skip unnecessary tags, do not repeat the code, output just the doc string."
    # ]

    # asking = [
    #     "Add doc-strings in English before the C++ method. Code: ",
    #     code_chunks[0],
    #     "use necessary tags, like @brief, @details @throws, @note, @param and @return, skip unnecessary tags",
    #     "Do not repeat the code." 
    # ]


    # asking = [
    #     "Add doc-strings in English before the C++ method. Code: ",
    #     code_chunks[0]
    #     ]


    # asking = [
    #     "Add English-language doxygen-style before the C++ method. Code: ",
    #     code_chunks[0]
    #     ]


    # asking

    # asking = [
    #     "Generate doc-strings including @brief, @details @throws, @note, @param, @return or other necessary tags for С++ method.",
    #     "Do NOT include any other text, analysis or explanations - ONLY the docstring.",
    #     "Method code:",
    #     code_chunks[0]
    # ]

    # asking = [
    #     "Below is a C++ method. Write its documentation string using doxygen-style tags.",
    #     "Include only these tags: @brief, @param, @return, @throws, @note as needed.",
    #     "Do NOT include any other text, analysis or explanations - ONLY the docstring.",
    #     "Method code:",
    #     code_chunks[0]
    # ]

    # asking = [
    #     "Generate C++ docstring in English for this method using EXACTLY this template:",
    #     "/**",
    #     " * @brief [one-line description]",
    #     " * @param [name] [description]",
    #     " * @return [description]",
    #     " * @throws [exception] [reason]",
    #     " * @note [optional note]",
    #     " */",
    #     "Method code: " + code_chunks[0],
    #     "STRICTLY follow the template above, skip unnecessary tags."
    # ]

    # asking = [
    #     "Generate ONLY the C++ docstring in English for this method. Do NOT include any analysis, explanations or additional text.",
    #     "STRICTLY follow these rules:",
    #     "1. Use ONLY these tags: @brief, @details, @throws, @note, @param, @return",
    #     "2. NEVER repeat information",
    #     "3. Output MUST be only the docstring itself without any other text",
    #     "4. Keep it concise and professional",
    #     "Method code: " + code_chunks[0]
    # ]
    # asking = [
    #     "Add doc-strings in English before the C++ method.",
    #     "Use tags: @brief, @param, @return, @throws, @note",
    #     "Code:",
    #     "void ADC4x250::write_adcCmpltInt(Tango::WAttribute &attr)",
    #     "{",
    #     "DEBUG_STREAM << \"ADC4x250::write_adcCmpltInt(Tango::WAttribute &attr) entering... \" << std::endl;",
    #     "//	Retrieve write value",
    #     "Tango::DevBoolean	w_val;",
    #     "attr.get_write_value(w_val);",
    #     "/----- PROTECTED REGION ID(ADC4x250::write_adcCmpltInt) ENABLED START -----/",
    #     "uint32_t int_ena = adc->get_int_enable();",
    #     "if(w_val)",
    #     "int_ena |= ADC4X250::ADC4X250::INT_ADC;",
    #     "else",
    #     "int_ena &= ~ADC4X250::ADC4X250::INT_ADC;",
    #     "adc->set_int_enable(int_ena);",
    #     "/----- PROTECTED REGION END -----/	//	ADC4x250::write_adcCmpltInt",
    #     "}"
    # ]
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

    # for idx, q in enumerate(query):
    #     # ans = llm.ask_question(q)
    #     _inputs, _count = llm.tokenize_input(q)
    #     # ans_tkn_count = max(2000, _count)
    #     ans_tkn_count = 1000
    #     ans = llm.ask_tokenized_question(_inputs, _count, ans_tkn_count)
    #     print("==========================================")
    #     print(ans)
    #     print("##########################################")
    #     print(extract_docstrings(ans))
    #     if idx != len(query)-1:
    #         txt = input("enter something to continue...")


    llm.finalize()




