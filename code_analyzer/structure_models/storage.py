import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class JsonDataStorage:
    """Implementation of data storage that saves to JSON Lines file and tracks statistics."""
    
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self._stats = {
            "classes": 0,
            "class_templates": 0,
            "functions": 0,
            "templates": 0,
            "namespaces": 0,
            "lambdas": 0,
            "macros": 0,
            "preprocessor_directives": 0,
            "literals": 0,
            "attributes": 0,
            "error_handlers": 0
        }
        
        # Data storage containers
        self.classes: Dict[str, Dict] = {}
        self.class_templates: Dict[str, Dict] = {}
        self.functions: List[Dict] = []
        self.templates: List[Dict] = []
        self.namespaces: List[Dict] = []
        self.lambdas: List[Dict] = []
        self.macros: List[Dict] = []
        self.preprocessor_directives: List[Dict] = []
        self.error_handlers: List[Dict] = []
        self.literals: List[Dict] = []
        self.attributes: List[Dict] = []

    def add_class(self, class_data: Dict) -> None:
        """Add a class to storage."""
        class_name = class_data["name"]
        if class_name not in self.classes:
            self.classes[class_name] = class_data
            self._stats["classes"] += 1

    def add_class_template(self, template_data: Dict) -> None:
        """Add a class template to storage."""
        template_name = template_data["name"]
        if template_name not in self.class_templates:
            self.class_templates[template_name] = template_data
            self._stats["class_templates"] += 1

    def add_function(self, function_data: Dict) -> None:
        """Add a function to storage."""
        self.functions.append(function_data)
        self._stats["functions"] += 1

    def add_function_template(self, template_data: Dict) -> None:
        """Add a function template to storage."""
        self.templates.append(template_data)
        self._stats["templates"] += 1

    def add_namespace(self, namespace_data: Dict) -> None:
        """Add a namespace to storage."""
        self.namespaces.append(namespace_data)
        self._stats["namespaces"] += 1

    def add_lambda(self, lambda_data: Dict) -> None:
        """Add a lambda to storage."""
        self.lambdas.append(lambda_data)
        self._stats["lambdas"] += 1

    def add_macro(self, macro_data: Dict) -> None:
        """Add a macro to storage."""
        self.macros.append(macro_data)
        self._stats["macros"] += 1

    def add_preprocessor_directive(self, directive_data: Dict) -> None:
        """Add a preprocessor directive to storage."""
        self.preprocessor_directives.append(directive_data)
        self._stats["preprocessor_directives"] += 1

    def add_literal(self, literal_data: Dict) -> None:
        """Add a user-defined literal to storage."""
        self.literals.append(literal_data)
        self._stats["literals"] += 1

    def add_attribute(self, attribute_data: Dict) -> None:
        """Add an attribute to storage."""
        self.attributes.append(attribute_data)
        self._stats["attributes"] += 1

    def add_error_handler(self, handler_data: Dict) -> None:
        """Add an error handler to storage."""
        self.error_handlers.append(handler_data)
        self._stats["error_handlers"] += 1

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics."""
        return self._stats.copy()

    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all collected data as a single list."""
        results = []
        results.extend(cls for cls in self.classes.values() if cls["methods"])
        results.extend(cls_tmpl for cls_tmpl in self.class_templates.values() if cls_tmpl["methods"])
        results.extend(self.functions)
        results.extend(self.templates)
        results.extend(self.namespaces)
        results.extend(self.lambdas)
        results.extend(self.error_handlers)
        results.extend(self.macros)
        results.extend(self.preprocessor_directives)
        results.extend(self.literals)
        results.extend(self.attributes)
        return results

    def flush(self) -> None:
        """Flush any buffered data to storage."""
        pass  # No buffering in this implementation

    def save_to_file(self) -> None:
        """Save all collected data to JSON Lines file."""
        if not self.output_path:
            return

        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for item in self.get_all_data():
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"[ERROR] Failed to save data to {self.output_path}: {e}")

    def print_statistics(self, unprocessed_stats: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """Print detailed statistics about collected data."""
        print("\n=== Extraction Statistics ===")
        print(f"Classes: {self._stats['classes']}")
        print(f"Class Templates: {self._stats['class_templates']}")
        print(f"Functions: {self._stats['functions']}")
        print(f"Function Templates: {self._stats['templates']}")
        print(f"Namespaces: {self._stats['namespaces']}")
        print(f"Lambdas: {self._stats['lambdas']}")
        print(f"Macros: {self._stats['macros']}")
        print(f"Preprocessor Directives: {self._stats['preprocessor_directives']}")
        print(f"User-defined Literals: {self._stats['literals']}")
        print(f"Attributes: {self._stats['attributes']}")
        print(f"Error Handlers: {self._stats['error_handlers']}")

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