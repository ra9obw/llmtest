import unittest
from enum import Enum
from cxx_instruction_generator import (
    GenerationMode,
    generate_instruction
)

class TestCppInstructionGenerator(unittest.TestCase):
    def setUp(self):
        self.function_element = {
            "type": "function_decl",
            "name": "calculateSum",
            "parent_name": ""
        }
        self.method_element = {
            "type": "cxx_method",
            "name": "getValue",
            "parent_name": "MyClass"
        }
        self.constructor_element = {
            "type": "constructor",
            "name": "",  # для конструктора имя не используется
            "parent_name": "MyClass"
        }
        self.destructor_element = {
            "type": "destructor",
            "name": "",  # для деструктора имя не используется
            "parent_name": "MyClass"
        }

    # Тесты для функции
    def test_function_full(self):
        result = generate_instruction(self.function_element, GenerationMode.FULL)
        # print(result)
        self.assertEqual(result, "Implement the C++ function calculateSum")

    def test_function_first_fragment(self):
        result = generate_instruction(self.function_element, GenerationMode.FIRST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the first fragment of the C++ function calculateSum")

    def test_function_next_fragment(self):
        result = generate_instruction(self.function_element, GenerationMode.NEXT_FRAGMENT, 2)
        # print(result)
        self.assertEqual(result, "Implement fragment (#2) of the C++ function calculateSum")

    def test_function_last_fragment(self):
        result = generate_instruction(self.function_element, GenerationMode.LAST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the last fragment of the C++ function calculateSum")

    # Тесты для метода
    def test_method_full(self):
        result = generate_instruction(self.method_element, GenerationMode.FULL)
        # print(result)
        self.assertEqual(result, "Implement the method getValue of the C++ class MyClass")

    def test_method_first_fragment(self):
        result = generate_instruction(self.method_element, GenerationMode.FIRST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the first fragment of the method getValue of the C++ class MyClass")

    def test_method_next_fragment(self):
        result = generate_instruction(self.method_element, GenerationMode.NEXT_FRAGMENT, 3)
        # print(result)
        self.assertEqual(result, "Implement the next fragment (#3) of the method getValue of the C++ class MyClass")

    def test_method_last_fragment(self):
        result = generate_instruction(self.method_element, GenerationMode.LAST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the last fragment of the method getValue of the C++ class MyClass")

    # Тесты для конструктора
    def test_constructor_full(self):
        result = generate_instruction(self.constructor_element, GenerationMode.FULL)
        # print(result)
        self.assertEqual(result, "Implement the constructor of the C++ class MyClass")

    def test_constructor_first_fragment(self):
        result = generate_instruction(self.constructor_element, GenerationMode.FIRST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the first fragment of the constructor of the C++ class MyClass")

    def test_constructor_next_fragment(self):
        result = generate_instruction(self.constructor_element, GenerationMode.NEXT_FRAGMENT, 1)
        # print(result)
        self.assertEqual(result, "Implement the next fragment (#1) of the constructor of the C++ class MyClass")

    def test_constructor_last_fragment(self):
        result = generate_instruction(self.constructor_element, GenerationMode.LAST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the last fragment of the constructor of the C++ class MyClass")

    # Тесты для деструктора
    def test_destructor_full(self):
        result = generate_instruction(self.destructor_element, GenerationMode.FULL)
        # print(result)
        self.assertEqual(result, "Implement the destructor of the C++ class MyClass")

    def test_destructor_first_fragment(self):
        result = generate_instruction(self.destructor_element, GenerationMode.FIRST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the first fragment of the destructor of the C++ class MyClass")

    def test_destructor_next_fragment(self):
        result = generate_instruction(self.destructor_element, GenerationMode.NEXT_FRAGMENT, 4)
        # print(result)
        self.assertEqual(result, "Implement the next fragment (#4) of the destructor of the C++ class MyClass")

    def test_destructor_last_fragment(self):
        result = generate_instruction(self.destructor_element, GenerationMode.LAST_FRAGMENT)
        # print(result)
        self.assertEqual(result, "Implement the last fragment of the destructor of the C++ class MyClass")

    # Тест на неизвестный тип
    def test_unknown_type(self):
        unknown_element = {
            "type": "unknown_type",
            "name": "test",
            "parent_name": "TestClass"
        }
        with self.assertRaises(NotImplementedError):
            generate_instruction(unknown_element, GenerationMode.FULL)

if __name__ == '__main__':
    unittest.main()