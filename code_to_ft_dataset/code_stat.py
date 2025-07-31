import json
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def plot_single_histograms(stat):
    """Функция для построения гистограмм длин кода по типам элементов"""
    for elem_type, data in stat.items():
        # Пропускаем служебные поля
        lengths = [v["length"] for k, v in data.items() 
                  if k not in ["max_desc", "min_desc", "max_len", "min_len"]]
        
        if not lengths:
            continue
            
        plt.figure(figsize=(12, 7))
        n_bins = 50
        _, bins, _ = plt.hist(lengths, bins=n_bins, alpha=0.7, color='blue', log=True)
        
        # Вычисляем размер бина (берем первый интервал как пример)
        bin_size = bins[1] - bins[0]
        
        plt.title(f'{elem_type}\n'
                 f'Total elements: {len(lengths)}, Bin size: {bin_size:.1f} chars')
        plt.xlabel('Code length (characters)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Добавляем вертикальные линии с полным описанием
        plt.axvline(x=data["min_len"], color='red', linestyle='--', 
                   label=f'Min: {data["min_len"]} chars\n{data["min_desc"]}')
        plt.axvline(x=data["max_len"], color='green', linestyle='--', 
                   label=f'Max: {data["max_len"]} chars\n{data["max_desc"]}')
        
        # Настраиваем легенду с переносом текста
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
        plt.tight_layout()
        
        plt.show()

def plot_histograms(stat):
    """Функция для построения гистограмм длин кода по типам элементов в одном окне"""
    eltype_to_plot = ["classes", "class_templates", "functions", "methods"]
    
    # Создаем фигуру с 4 субплогами (2x2)
    plt.figure(figsize=(16, 12))
    
    for idx, elem_type in enumerate(eltype_to_plot, 1):
        data = stat[elem_type]
        # Пропускаем служебные поля
        lengths = [v["length"] for k, v in data.items() 
                  if k not in ["max_desc", "min_desc", "max_len", "min_len"]]
        
        if not lengths:
            continue
            
        # Выбираем текущий субплот
        ax = plt.subplot(2, 2, idx)
        n_bins = 50
        _, bins, _ = ax.hist(lengths, bins=n_bins, alpha=0.7, color='blue', log=True)
        
        # Вычисляем размер бина
        bin_size = bins[1] - bins[0]
        
        ax.set_title(f'{elem_type}\n'
                     f'Total elements: {len(lengths)}, Bin size: {bin_size:.1f} chars')
        ax.set_xlabel('Code length (characters)')
        ax.set_ylabel('Frequency (log scale)')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        # Добавляем вертикальные линии
        ax.axvline(x=data["min_len"], color='red', linestyle='--', 
                  label=f'Min: {data["min_len"]} chars\n{data["min_desc"]}')
        ax.axvline(x=data["max_len"], color='green', linestyle='--', 
                  label=f'Max: {data["max_len"]} chars\n{data["max_desc"]}')
        ax.axvline(x=512, color='black', linestyle='--', 
                  label=f'512 tokens limit')
        
        # Настраиваем легенду
        ax.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center', borderaxespad=0.)
    
    # Регулируем отступы между графиками
    plt.tight_layout()
    plt.show()

def clean_cpp_code(code):
    # Удаляем комментарии
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # /* ... */
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)  # // ...
    code = re.sub(r'///.*?$', '', code, flags=re.MULTILINE)  # /// ...
    code = re.sub(r'//!.*?$', '', code, flags=re.MULTILINE)  # //! ...

    # Нормализация пробелов:
    # 1. Заменяем все последовательности пробелов/табов на один пробел
    code = re.sub(r'[ \t]+', ' ', code)
    # 2. Убираем пробелы перед <, >, ::, *, &, (, ), {, }, [, ], ;, ,, ::
    code = re.sub(r' ([<>{}\[\]();,&*:]|::)', r'\1', code)
    # 3. Убираем пробелы после <, >, ::, *, &, (, {, [ 
    code = re.sub(r'([<>{}\[(::*&]) ', r'\1', code)
    # 4. Убираем пробелы в начале/конце строки
    code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())

    return code

def main():
    # from settings import settings
    from transformers import (
    AutoTokenizer
    )

    BASE_ROOT = r"C:\\work\\llm_test"
    # BASE_ROOT = settings["BASE_ROOT"]
    # PROJ_NAME = settings["PROJ_NAME"]
    # PROJ_NAME = r"simple"
    # PROJ_NAME = r"adc4x250"
    PROJ_NAME = r"cppTango-9.3.7"
    # PROJ_NAME = r"test_examples"
    INPUT_JSONL = os.path.join(BASE_ROOT, f"dataset_clang_{PROJ_NAME}.jsonl")

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Нужно для padding

    stat = {}

    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f:
        for line in in_f:
            entry = json.loads(line)
            _code = entry.get("code", "")
            _desc = f"{entry['name']} in {entry['location']}:{entry['line']}"

            # _len = len(_code)
            _code = clean_cpp_code(_code)
            _input_tokens_no_trunc = tokenizer(_code, truncation=False)
            _len = len(_input_tokens_no_trunc[0])
            
            if entry['type'] not in stat:
                stat[entry['type']] = {
                    "max_desc": _desc, 
                    "min_desc": _desc, 
                    "max_len": _len, 
                    "min_len": _len, 
                    entry["name"]: {
                        "length": _len, 
                        "lines": len(_code.split("\n")), 
                        "lines_length": [len(x) for x in _code.split("\n")]
                    }
                }
            else:
                
                stat_type = stat[entry['type']]
                
                if _len > stat_type["max_len"]:
                    stat_type["max_desc"] = _desc
                    stat_type["max_len"] = _len
                if _len < stat_type["min_len"]:
                    stat_type["min_desc"] = _desc
                    stat_type["min_len"] = _len
                    
                stat_type[entry["name"]] = {
                    "length": _len, 
                    "lines": len(_code.split("\n")), 
                    "lines_length": [len(x) for x in _code.split("\n")]
                }
                
    for k, v in stat.items():
        print(f"for {k} type\tmin len is\t{v['min_len']}\t{v['min_desc']}\n"
              f"and max len is {v['max_len']}\t{v['max_desc']}\n")
    
    plot_single_histograms(stat)

if __name__ == "__main__":
    main()