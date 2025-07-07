import os
import json
import subprocess
from pathlib import Path

# Путь к локальному репозиторию
# REPO_PATH = Path("E:\\work\\llm_test\\codebase\\cppTango-9.3.7")
REPO_PATH = Path(r"C:\\work\\llm_test\\codebase\\simple")
OUTPUT_JSONL = Path("C:\\work\\llm_test\\dataset.jsonl")

# одна запись - один файл

# Расширения C++ файлов
CXX_EXTENSIONS = {".cpp", ".cxx", ".cc", ".c++", ".h", ".hh", ".hpp", ".hxx"}

# Параметры clang-format
CLANG_FORMAT_CMD = ["clang-format", "-style=LLVM", "-assume-filename={filename}"]

def is_cxx_file(file_path):
    return file_path.suffix.lower() in CXX_EXTENSIONS

def format_code_with_clang(code, filename):
    # Запускаем clang-format на основе содержимого строки
    process = subprocess.run(
        CLANG_FORMAT_CMD[0],  # команда
        input=code,
        text=True,
        capture_output=True,
        check=False,
        env={"FILENAME": filename}
    )
    if process.returncode == 0:
        return process.stdout
    else:
        print(f"[!] Clang-format error for {filename}: {process.stderr}")
        return code  # Возвращаем оригинальный код

def walk_and_process(repo_path):
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = Path(root) / file
                if is_cxx_file(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            original_code = f.read()
                        # Форматируем код
                        formatted_code = format_code_with_clang(original_code, str(file_path))
                        # Сохраняем в JSONL
                        entry = {
                            "file": str(file_path.relative_to(repo_path)),
                            "code": formatted_code.strip(),
                        }
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"[!] Error processing {file_path}: {e}")

if __name__ == "__main__":
    walk_and_process(REPO_PATH)
    print(f"[+] Предобработка завершена. Результирующий датасет: {OUTPUT_JSONL}")