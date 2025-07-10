import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

class JsonDataStorage:
    """Implementation of data storage that saves to JSON Lines file and tracks statistics."""
    
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self._index = {}  # Для быстрого поиска элементов
        self.data = {
            "classes": [],
            "class_templates": [],
            "functions": [],
            "function_templates": [], 
            "methods": [],
            "template_methods": [],
            "namespaces": [],
            "macros": [],
            "attributes": [],
            "error_handlers": [],
            "lambdas": [],
            "literals": [],
            "preprocessor_directives": []
        }
        # Data storage containers

    def add_element(self, element_type: str, element_data: Dict[str, Any]) -> None:
            """Добавляет элемент в соответствующую коллекцию."""
            if element_type not in self.data:
                raise ValueError(f"Unknown element type: {element_type}")
            
            self.data[element_type].append(element_data)
            self._update_index(element_type, element_data)

    def get_or_create_id(self, element_type: str, match_fields: Dict[str, Any]) -> str:
        """Находит или создает элемент и возвращает его ID."""
        if existing := self.find_element(element_type, match_fields):
            return existing["id"]
        
        new_id = str(uuid.uuid4())
        new_element = {"id": new_id, **match_fields}
        self.add_element(element_type, new_element)
        return new_id

    def find_element(self, element_type: str, match_fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ищет элемент по заданным полям."""
        return next(
            (item for item in self.data.get(element_type, []) 
            if all(item.get(k) == v for k, v in match_fields.items())),
            None
        )

    def _update_index(self, element_type: str, element_data: Dict[str, Any]) -> None:
        """Обновляет индекс для быстрого поиска."""
        if "id" not in element_data:
            return
            
        self._index[element_data["id"]] = (element_type, len(self.data[element_type]) - 1)

    def get_by_id(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Получает элемент по ID."""
        if element_id not in self._index:
            return None
            
        element_type, index = self._index[element_id]
        return self.data[element_type][index]
    
    # def get_all_data(self) -> List[Dict[str, Any]]:
    #     """Get all collected data as a single list."""
    #     results = []
    #     results.extend(cls for cls in self.classes.values() if cls["methods"])
    #     results.extend(cls_tmpl for cls_tmpl in self.class_templates.values() if cls_tmpl["methods"])
    #     results.extend(self.functions)
    #     results.extend(self.templates)
    #     results.extend(self.namespaces)
    #     results.extend(self.lambdas)
    #     results.extend(self.error_handlers)
    #     results.extend(self.macros)
    #     results.extend(self.preprocessor_directives)
    #     results.extend(self.literals)
    #     results.extend(self.attributes)
    #     return results

    def flush(self) -> None:
        """Flush any buffered data to storage."""
        pass  # No buffering in this implementation

    def save_to_file(self) -> None:
        """Сохраняет данные в файл в формате JSONL."""
        if not self.output_path:
            return

        with open(self.output_path, "w", encoding="utf-8") as f:
            for element_type, elements in self.data.items():
                for element in elements:
                    json_line = json.dumps({
                        "type": element_type,
                        "data": element
                    }, ensure_ascii=False)
                    f.write(json_line + "\n")

    def print_statistics(self, unprocessed_stats: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """Print detailed statistics about collected data."""
        for element_type, elements in self.data.items():
            print(f"{element_type}: {len(elements)}")

        if unprocessed_stats:
            print("\n=== Unprocessed Elements ===")
            if unprocessed_stats.get("unprocessed_unexpected"):
                print("\nUnprocessed cursor kinds:")
                for kind, count in sorted(
                    unprocessed_stats["unprocessed_unexpected"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ):
                    print(f"{kind}: {count}")
            else:
                print("\nAll cursor kinds were processed")