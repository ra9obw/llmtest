import json
import os
from pathlib import Path
from docstring_promt_generator import get_docstring
from docstring_promt_generator import  get_docstring_batch
# def get_docstring_batch(
#     llm,
#     requests: list  # list of dicts with same params as get_docstring
# ) -> list:
#     print("get_docstring_batch")
#     print(requests)
#     print(len(requests))
#     return len(requests)*['']

class DocstringAdder:
    def __init__(self, input_file: str, output_file: str, llm=None):
        self.llm = llm
        self.input_file = input_file
        self.output_file = output_file
        self.temp_file = f"{output_file}.tmp"
        self.processed_ids = set()
        self.current_output = None
        self.last_position = 0

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

        self.current_output = open(self.temp_file, 
                                 'a+' if Path(self.temp_file).exists() else 'w+', 
                                 encoding='utf-8')
        self.last_position = self.current_output.tell()

    def _save_item(self, item: dict):
        """Save single item to output, replacing previous version if type and name match"""
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
        
        if found_index >= 0:
            del lines[found_index]
        
        lines.append(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.current_output.seek(0)
        self.current_output.truncate()
        self.current_output.writelines(lines)
        self.current_output.flush()
        os.fsync(self.current_output.fileno())

    def _group_methods_by_length(self, methods: list) -> dict:
        """Group methods by their code length"""
        groups = {
            'short': [],   # 0-100 chars
            'medium': [],  # 100-300 chars
            'long': []     # 300+ chars
        }
        print("_group_methods_by_length")
        for method in methods:
            code_length = len(method.get('code', ''))
            print(f"{method['name']}\tcode_length =\t{code_length}")
            if code_length <= 1500:
                groups['short'].append(method)
            elif code_length <= 3000:
                groups['medium'].append(method)
            else:
                groups['long'].append(method)
        
        for k,v in groups.items():
            print(f"{k}:\t{len(v)}")
        return groups

    def _process_method_batch(self, methods: list, class_item: dict, batch_size: int):
        """Process a batch of methods and update class item"""
        # Prepare batch requests
        batch_requests = []
        for method in methods:
            request = {
                'name': method['name'],
                'item_type': 'method',
                'signature': method.get('signature', {}),
                'code': method.get('code', '')
            }
            if class_item['type'] == 'class_template':
                request['item_type'] = 'template_method'
                request['template_params'] = method.get('template_parameters', [])
            
            batch_requests.append(request)
        
        # Process in batches
        for i in range(0, len(batch_requests), batch_size):
            current_batch = batch_requests[i:i+batch_size]
            docstrings = get_docstring_batch(llm=self.llm, requests=current_batch)
            
            # Update methods with docstrings
            for j, docstring in enumerate(docstrings):
                method_idx = i + j
                if method_idx < len(methods):
                    methods[method_idx]['docstring'] = docstring
        
        return methods

    def _add_class_docstrings(self, class_item: dict) -> dict:
        """Add docstrings to class and its methods using batch processing"""
        # Add class-level docstring
        # class_item['docstring'] = get_docstring(
        #     llm=self.llm,
        #     name=class_item['name'],
        #     item_type='class',
        #     declaration=class_item.get('declaration', ''),
        #     methods=class_item.get('methods', [])
        # )

        # Group methods by length and process in batches
        method_groups = self._group_methods_by_length(class_item.get('methods', []))
        
        # Process each group with appropriate batch size
        class_item['methods'] = []
        for group, methods in method_groups.items():
            if not methods:
                continue
                
            if group == 'short':
                batch_size = 4
            elif group == 'medium':
                batch_size = 2
            else:  # long
                batch_size = 1
                
            processed_methods = self._process_method_batch(methods, class_item, batch_size)
            class_item['methods'].extend(processed_methods)
            
            # Save after each group is processed
            self._save_item(class_item)

        return class_item

    def _add_template_class_docstrings(self, class_item: dict) -> dict:
        """Add docstrings to template class and its methods using batch processing"""
        class_item['docstring'] = get_docstring(
            llm=self.llm,
            name=class_item['name'],
            item_type='template_class',
            template_params=class_item.get('template_parameters', []),
            methods=class_item.get('methods', [])
        )

        # Group methods by length and process in batches
        method_groups = self._group_methods_by_length(class_item.get('methods', []))
        
        # Process each group with appropriate batch size
        class_item['methods'] = []
        for group, methods in method_groups.items():
            if not methods:
                continue
                
            if group == 'short':
                batch_size = 6
            elif group == 'medium':
                batch_size = 4
            else:  # long
                batch_size = 2
                
            processed_methods = self._process_method_batch(methods, class_item, batch_size)
            class_item['methods'].extend(processed_methods)
            
            # Save after each group is processed
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
                        
                        if 'id' in item and item['id'] in self.processed_ids:
                            continue

                        if item['type'] == 'class':
                            item = self._add_class_docstrings(item)
                        elif item['type'] == 'class_template':
                            item = self._add_template_class_docstrings(item)
                        else:
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
    processor = DocstringAdder(input_file, output_file)
    processor.process()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.1f} seconds")