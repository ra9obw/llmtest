import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from interfaces import IJsonDataStorage
import uuid

class JsonDataStorage(IJsonDataStorage):
    """Implementation of data storage that saves to JSON Lines file and tracks statistics."""
    
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self.data = {}
        
    def create_element_storage(self, fields: Set[str]) -> None:
        for field_name in fields:
            self.data[field_name] = []

    def add_element(self, element_type: str, element_data: Dict[str, Any]) -> None:
            """Добавляет элемент в соответствующую коллекцию."""
            if element_type not in self.data:
                raise ValueError(f"Unknown element type: {element_type}")
            
            self.data[element_type].append(element_data)

    def save_to_file(self) -> None:
        """Сохраняет данные в файл в формате JSONL."""
        if not self.output_path:
            return

        with open(self.output_path, "w", encoding="utf-8") as f:
            for _, elements in self.data.items():
                for element in elements:
                    json_line = json.dumps(element, ensure_ascii=False)
                    f.write(json_line + "\n")

    def print_statistics(self, unprocessed_stats: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """Print detailed statistics about collected data."""
        for element_type, elements in self.data.items():
            print(f"{element_type}: {len(elements)}")

        if unprocessed_stats:
            print("\n=== Unprocessed Elements ===")
            if unprocessed_stats.get("expected"):
                print("\nUnprocessed expected cursor kinds:")
                for kind, count in sorted(
                    unprocessed_stats["expected"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ):
                    print(f"{kind}: {count}")
            if unprocessed_stats.get("unexpected"):
                print("\nUnprocessed unexpected cursor kinds:")
                for kind, count in sorted(
                    unprocessed_stats["unexpected"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ):
                    print(f"{kind}: {count}")
            else:
                print("\nAll cursor kinds were processed")