
import os

import re
import csv
from pathlib import Path

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

def parse_method_docs(file_content):
    method_dict = {}

    # Шаблон для поиска строк с объявлениями методов
    method_pattern = re.compile(r"Class\s+'(\w+)'\s+:\s+method\s+(\w+)")

    # Разделяем содержимое файла на строки
    lines = file_content.split('\n')
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        # Ищем строку с объявлением метода
        match = method_pattern.search(line)
        if match:
            class_name = match.group(1)
            method_name = match.group(2)
            i += 1

            # Проверяем следующую строку на наличие []
            if i < n and lines[i].strip() == '[]':
                method_dict[method_name] = ''
                i += 1
            else:
                # Собираем все строки документации до следующего объявления метода
                doc_lines = []
                while i < n:
                    next_line = lines[i].strip()
                    # Проверяем, не начинается ли следующая строка с нового объявления метода
                    if method_pattern.search(next_line):
                        break
                    doc_lines.append(next_line)
                    i += 1
                # Объединяем строки документации
                # method_dict[method_name] = extract_docstrings('\n'.join(doc_lines).strip())
                method_dict[method_name] = '\n'.join(doc_lines).strip()
        else:
            i += 1

    return method_dict


def read_and_parse_file(file_path):
    # try:
    #     with open(file_path, 'r', encoding='latin-1') as file:
    #         content = file.read()
    #     return parse_method_docs(content)
    # except FileNotFoundError:
    #     print(f"Ошибка: файл {file_path} не найден")
    #     return {}
    # except Exception as e:
    #     print(f"Произошла ошибка при чтении файла: {e}")
    #     return {}
    encodings_to_try = ['utf-16', 'latin-1',]

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as file:
                content = file.read()
            return parse_method_docs(content)
        except UnicodeError:
            continue

    print(f"Не удалось прочитать файл {file_path} ни в одной из кодировок")
    return {}


def save_results_to_file(result, _len, filename="results.txt"):
    """
    Сохраняет результаты в текстовый файл в заданном формате.

    :param result: Список словарей с результатами для каждого документа
    :param _len: Количество документов
    :param filename: Имя файла для сохранения (по умолчанию "results.txt")
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for method_name in result[0].keys():
            file.write(f"Метод: {method_name}\n")
            for i in range(0, _len):
                file.write(f'Doc_File_{i}\n')
                file.write(f"{result[i].get(method_name, '')}\n")
                file.write("-" * 50 + "\n")
            file.write("#" * 50 + "\n")

# Пример использования
if __name__ == "__main__":
    DOC_ROOT_PATH = [
        # C:\work\llm_test\doc_2.txt
                    #  r"С:\\цщкл\\llm_еesе\\doc_1.txt",
                     r"C:\\work\\llm_test\\doc_2.txt",
                     r"C:\\work\\llm_test\\doc_3.txt",
                     r"C:\\work\\llm_test\\doc_4.txt",
                    #  r"С:\\цщкл\\llm_еesе\\doc_5.txt"
                     ]
    # Укажите путь к вашему файлу
    result = [read_and_parse_file(p) for p in DOC_ROOT_PATH]
    output_path = Path(DOC_ROOT_PATH[0]).parent / "methods_documentation.csv"

    save_results_to_file(result, len(result), output_path)
    # result = []
    # for file_path in DOC_ROOT_PATH:
    #     # Читаем файл и парсим его содержимое
    #     result.append(read_and_parse_file(file_path))
    #
    # _len = len(DOC_ROOT_PATH)
    #
    #
    # output_path = Path(DOC_ROOT_PATH[0]).parent / "methods_documentation.csv"
    # fieldnames = ['Method'] + [f'Doc_File_{i}' for i in range(0, _len)]
    #
    # with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    #
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     for method,v in result[0].items():
    #         row = {'Method': method}
    #         for i in range(0, _len):
    #             row[f'Doc_File_{i}'] = result[i].get(method,'')
    #         writer.writerow(row)

    # # Выводим результат
    # for method_name in result[0].keys():
    #     print(f"Метод: {method_name}")
    #     for i in range(0, _len):
    #         print(f'Doc_File_{i}')
    #
    #     # row[f'Doc_File_{i}'] = result[i].get(method,'')
    #         print(result[i].get(method_name,''))
    #         print("#" * 50)

