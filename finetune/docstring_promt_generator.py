from llm_promt_driver import llm_promt_driver
# from llm_promt_driver_via_server import LLMPromptDriver as llm_promt_driver
import re
import json

def get_docstring(
    llm,
    name: str,
    item_type: str,  # 'class', 'method', 'template_class', etc.
    declaration: str = '',
    signature: dict = None,
    code: str = '',
    template_params: list = None,
    methods: list = None
) -> str:
    
    if item_type == "method":

        print(f"docstring for {name} with {item_type} type")

        with llm_promt_driver() as driver: 
            return generate_docstring(code, driver)

    else:
        return ""


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

def generate_docstring(code, llm):

    asking = [
        "Add doc-strings in English before the C++ method.",
        "Use tags: @brief, @param, @return, @throws, @note",
        "Code:",
        code,
        "Use this template:",
        "/**",
        " * @brief [short description]",
        " * @param [name] [description]",
        " * @return [description]",
        " * @throws [exception] [reason]",
        " * @note [optional note]",
        " */",
        "skip @param, if there is no params",
        "for the @throws tag use exceptions that are clearly present in the code, skip @throws if there is no exceptions",
        "do not add description if function return void",
        "do not repeat the code after docstring"
    ]
    query = ["\n".join(asking)]
    return extract_docstrings(llm.ask_question(query[0], ans_size=1024))

def get_docstring_batch(
    llm,
    requests: list  # list of dicts with same params as get_docstring
) -> list:
    """
    Generate docstrings for multiple functions in batch.
    
    Args:
        llm: LLM prompt driver instance
        requests: List of dictionaries containing parameters for docstring generation.
                 Each dict should have same parameters as get_docstring function:
                 name, item_type, declaration, signature, code, template_params, methods
                 
    Returns:
        list: List of generated docstrings in the same order as input requests
    """
    results = [""] * len(requests)
    method_requests = []
    method_indices = []
    
    for req in requests:
        print(f"req for {req.get('name')}")

    # Collect all method requests that need processing
    for i, request in enumerate(requests):
        if request.get('item_type') == "method":
            method_requests.append(request)
            method_indices.append(i)
    
    if not method_requests:
        return results
    
    # Prepare batch queries for LLM
    queries = []
    for request in method_requests:
        asking = [
            "Add doc-strings in English before the C++ method.",
            "Use tags: @brief, @param, @return, @throws, @note",
            "Code:",
            request.get('code', ''),
            "Use this template:",
            "/**",
            " * @brief [short description]",
            " * @param [name] [description]",
            " * @return [description]",
            " * @throws [exception] [reason]",
            " * @note [optional note]",
            " */",
            "skip @param, if there is no params",
            "for the @throws tag use exceptions that are clearly present in the code, skip @throws if there is no exceptions",
            "do not add description if function return void",
            "do not repeat the code after docstring"
        ]
        queries.append("\n".join(asking))
    
    # Process batch with LLM
    with llm_promt_driver() as driver:
        responses = driver.ask_batched_questions(queries, ans_size=1024)
    
    # Extract docstrings and fill results
    for idx, response in zip(method_indices, responses):
        results[idx] = extract_docstrings(response)
    
    return results

import os
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

def get_func_from_json(class_name = "ADC4x250", method_name = "read_adc"):
    input_file = "C:\\work\\llm_test\\dataset_clang_adc4x250.jsonl"
    with open(input_file, 'r', encoding='utf-8') as infile:
        # try:
        for line in infile:
            entry = json.loads(line)
            print(len(entry), entry['type'])
            if (entry['type'] == "class") and (entry['name'] == class_name):
                for method in entry.get("methods", []):
                    if method['name'] == method_name:
                        print("Method found!")
                        return method.get('code', '')
    return ""


if __name__ == "__main__":
    # code_chunks = read_codebase("C:\\work\\llm_test\\codebase", "test.cpp")
    # print(f"Прочитано {len(code_chunks)} файлов.")
    # llm = llm_promt_driver()
    # llm.initialize()
    # docstring = generate_docstring(code_chunks[0], llm)
    # print(docstring)

    code_chunk = get_func_from_json("ADC4x250", "init_device")
    llm = llm_promt_driver()
    llm.initialize()
    docstring = generate_docstring(code_chunk, llm)
    print(docstring)
