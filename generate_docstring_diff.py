import json
import os
from typing import Dict, Any, List, Optional

def read_source_lines(location: str, project_path: str, line: int, num_context: int = 3) -> Optional[List[str]]:
    """
    Читает строки исходного кода вокруг указанной строки.
    
    Args:
        location: Относительный путь к файлу
        project_path: Путь к корню проекта
        line: Номер строки (1-based)
        num_context: Количество строк контекста с каждой стороны
        
    Returns:
        Список строк или None, если файл не найден
    """
    full_path = os.path.join(project_path, location)
    if not os.path.exists(full_path):
        return None
    
    with open(full_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    start = max(0, line - num_context - 1)  # -1 для перевода в 0-based
    end = min(len(all_lines), line + num_context - 1)
    
    return [line.rstrip('\n') for line in all_lines[start:end]]

def generate_diff(method: Dict[str, Any], project_path: str) -> str:
    """
    Генерирует git diff для добавления/обновления докстроки метода с контекстом.
    
    Args:
        method: Словарь с информацией о методе из JSON
        project_path: Путь к корню проекта
        
    Returns:
        Строка с git diff для этого метода или пустая строка, если не удалось прочитать контекст
    """
    if not method.get('docstring'):
        return ""
    
    location = method['location']
    line = method['line']
    docstring = method['docstring'].strip()
    
    # Получаем контекстные строки
    context_lines = read_source_lines(location, project_path, line)
    if context_lines is None:
        print(f"Warning: Файл {location} не найден в проекте {project_path}, пропускаем")
        return ""
    
    # Разбиваем докстроку на строки и удаляем trailing whitespace
    docstring_lines = [line.rstrip() for line in docstring.split('\n')]
    num_doc_lines = len(docstring_lines)
    num_context = len(context_lines)
    
    # Формируем заголовок diff
    diff_header = f"""diff --git a/{location} b/{location}
index 0000000..0000000 100644
--- a/{location}
+++ b/{location}
"""
    # Формат: @@ -старая_строка,старые_строки +новая_строка,новые_строки @@
    # Включаем контекстные строки до и после
    chunk_header = f"@@ -{line - 3},{num_context} +{line - 3},{num_context + num_doc_lines} @@\n"
    
    # Собираем тело diff
    diff_body = []
    
    # Добавляем контекстные строки (с пробелом в начале)
    for ctx_line in context_lines[:3]:  # строки до
        diff_body.append(f" {ctx_line}")
    
    # Добавляем новые строки докстроки (с + в начале)
    for doc_line in docstring_lines:
        diff_body.append(f"+{doc_line}")
    
    # Добавляем контекстные строки после (с пробелом в начале)
    for ctx_line in context_lines[3:]:  # строки после
        diff_body.append(f" {ctx_line}")
    
    return diff_header + chunk_header + "\n".join(diff_body) + "\n"

def process_json_file(json_file_path: str, output_diff_path: str, project_path: str) -> None:
    """
    Обрабатывает JSON файл и генерирует итоговый diff файл.
    
    Args:
        json_file_path: Путь к входному JSON файлу
        output_diff_path: Путь к выходному diff файлу
        project_path: Путь к корню проекта
    """
    with open(json_file_path, "r", encoding="utf-8") as in_f, \
         open(output_diff_path, "w", encoding="utf-8", newline='\n') as out_f:
        
        for line in in_f:
            entry = json.loads(line)
            
            if entry['type'] == "method":
                diff = generate_diff(entry, project_path)
                if diff:
                    out_f.write(diff)
                    out_f.write("\n")
            elif entry['type'] == "class":
                for method in entry.get("methods", []):
                    diff = generate_diff(method, project_path)
                    if diff:
                        out_f.write(diff)
                        out_f.write("\n")

if __name__ == "__main__":
    import argparse

    # Значения по умолчанию
    _ifile = "dataset_clang_adc4x250_doc_2.jsonl"
    _ofile = "docstrings_adc4x250_doc_2.diff"
    _cb_path = "C:\\work\\llm_test\\codebase\\adc4x250"
    
    parser = argparse.ArgumentParser(description='Generate git diff for docstrings from JSON data')
    parser.add_argument('--input', '-i', dest='input_json', 
                       default=_ifile, help='Path to input JSON file')
    parser.add_argument('--output', '-o', dest='output_diff', 
                       default=_ofile, help='Path to output diff file')
    parser.add_argument('--project-path', '-p', dest='project_path',
                       default=_cb_path, help='Path to project root directory')
    
    args = parser.parse_args()
    
    print(f"Обработка файла: {args.input_json}")
    print(f"Путь к проекту: {args.project_path}")
    print(f"Результат будет записан в: {args.output_diff}")
    
    process_json_file(args.input_json, args.output_diff, args.project_path)
    print(f"Diff файл успешно сгенерирован: {args.output_diff}")