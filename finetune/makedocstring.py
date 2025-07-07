import json
import os
from pathlib import Path
from docstring_promt_generator import get_docstring
# from llm_promt_driver import llm_promt_driver
from llm_promt_driver_via_server import LLMPromptDriver as llm_promt_driver


class DocstringAdder:
    def __init__(self, input_file: str, output_file: str, llm = None):
        self.llm = llm
        self.input_file = input_file
        self.output_file = output_file
        self.temp_file = f"{output_file}.tmp"
        self.processed_ids = set()
        self.current_output = None
        self.last_position = 0  # Для отслеживания позиции в файле

    def _initialize_output(self):
        """Prepare output file and track processed items"""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"Resuming processing, found existing output: {self.output_file}")
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'id' in item:
                            self.processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        continue

        # Открываем файл для чтения и записи
        self.current_output = open(self.temp_file, 
                                 'a+' if Path(self.temp_file).exists() else 'w+', 
                                 encoding='utf-8')
        self.last_position = self.current_output.tell()

    def _save_item(self, item: dict):
        """Save single item to output, replacing previous version if type and name match"""
        # Проверяем, есть ли уже элемент с таким же типом и именем
        self.current_output.seek(0)
        lines = self.current_output.readlines()
        self.current_output.seek(0)
        
        found_index = -1
        for i, line in enumerate(lines):
            try:
                existing_item = json.loads(line)
                if (existing_item.get('type') == item.get('type') and 
                    existing_item.get('name') == item.get('name')):
                    found_index = i
                    break
            except json.JSONDecodeError:
                continue
        
        # Если нашли совпадение, удаляем старую запись
        if found_index >= 0:
            del lines[found_index]
        
        # Добавляем новую запись
        lines.append(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Перезаписываем весь файл
        self.current_output.seek(0)
        self.current_output.truncate()
        self.current_output.writelines(lines)
        self.current_output.flush()
        os.fsync(self.current_output.fileno())

    def _add_class_docstrings(self, class_item: dict) -> dict:
        """Add docstrings to class and its methods"""
        # Add class-level docstring
        class_item['docstring'] = get_docstring(
            llm=self.llm,
            name=class_item['name'],
            item_type='class',
            declaration=class_item.get('declaration', ''),
            methods=class_item.get('methods', [])
        )

        # Process each method and save after each one
        for method in class_item.get('methods', []):
            method['docstring'] = get_docstring(
                llm=self.llm,
                name=method['name'],
                item_type='method',
                signature=method.get('signature', {}),
                code=method.get('code', '')
            )
            
            # Save after each method is processed
            self._save_item(class_item)

        return class_item

    def _add_template_class_docstrings(self, class_item: dict) -> dict:
        """Add docstrings to template class and its methods"""
        class_item['docstring'] = get_docstring(
            llm=self.llm,
            name=class_item['name'],
            item_type='template_class',
            template_params=class_item.get('template_parameters', []),
            methods=class_item.get('methods', [])
        )

        for method in class_item.get('methods', []):
            method['docstring'] = get_docstring(
                llm=self.llm,
                name=method['name'],
                item_type='template_method',
                signature=method.get('signature', {}),
                template_params=method.get('template_parameters', []),
                code=method.get('code', '')
            )
            self._save_item(class_item)

        return class_item

    def process(self):
        """Main processing method"""
        self._initialize_output()

        try:
            with open(self.input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        item = json.loads(line)
                        
                        # Skip processed items
                        if 'id' in item and item['id'] in self.processed_ids:
                            continue

                        # Process based on type
                        if item['type'] == 'class':
                            item = self._add_class_docstrings(item)
                        elif item['type'] == 'class_template':
                            item = self._add_template_class_docstrings(item)
                        else:
                            # For non-class items, just save as-is
                            self._save_item(item)
                            continue

                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed line: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing item: {e}")
                        continue

        finally:
            if self.current_output:
                self.current_output.close()
            
            # Replace original file only if processing completed
            if os.path.exists(self.temp_file):
                os.replace(self.temp_file, self.output_file)
                print(f"Processing complete. Results saved to {self.output_file}")

if __name__ == "__main__":
    import time
    t1 = time.time()
    input_file = "C:\\work\\llm_test\\dataset_clang_adc4x250.jsonl"
    base = os.path.splitext(input_file)[0]
    output_file = f"{base}_doc.jsonl"

    print(f"Processing {input_file} -> {output_file}")
    # processor = DocstringAdder(input_file, output_file, llm)
    processor = DocstringAdder(input_file, output_file)
    processor.process()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.1f} секнуд на всё про всё")
