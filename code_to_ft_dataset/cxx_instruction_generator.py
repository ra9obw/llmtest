from enum import Enum, auto
import random

class GenerationMode(Enum):
    FULL = auto()
    FIRST_FRAGMENT = auto()
    NEXT_FRAGMENT = auto()
    LAST_FRAGMENT = auto()
    DOCSTRING = auto()

class CppInstructionGenerator:
    def __init__(self, element):
        self.element_name = element["name"]
        self.parent_name = element["parent_name"]
    
    def _gen_full_instruction(self):
        raise NotImplementedError()
    
    def _gen_first_fragment(self):
        raise NotImplementedError()

    def _gen_next_fragment(self, idx):
        raise NotImplementedError()

    def _gen_last_fragment(self):
        raise NotImplementedError()
        
    def _gen_docstring(self):
        raise NotImplementedError()

    def generate(self, mode: GenerationMode, idx="0"):
        if mode == GenerationMode.FULL:
            return self._gen_full_instruction()
        elif mode == GenerationMode.FIRST_FRAGMENT:
            return self._gen_first_fragment()
        elif mode == GenerationMode.LAST_FRAGMENT:
            return self._gen_last_fragment()
        elif mode == GenerationMode.DOCSTRING:
            return self._gen_docstring()
        else:
            return self._gen_next_fragment(idx)


instr_list = [
"Write",
"Create",
"Develop",
"Provide implementation of",
"Produce"
]
def randomInstruction():
    return random.choice(instr_list)


class FunctionTransformer(CppInstructionGenerator):
    def __init__(self, element):
        super().__init__(element)
    
    def _gen_full_instruction(self):
        return f"{randomInstruction()} the C++ function {self.element_name}"
    
    def _gen_first_fragment(self):
        return f"{randomInstruction()} the first fragment of the C++ function {self.element_name}"

    def _gen_next_fragment(self, idx):
        return f"{randomInstruction()} fragment (#{idx}) of the C++ function {self.element_name}"

    def _gen_last_fragment(self):
        return f"{randomInstruction()} the last fragment of the C++ function {self.element_name}"
        
    def _gen_docstring(self):
        return f"Write documentation (docstring) for the C++ function {self.element_name}"

class ClassTransformer(CppInstructionGenerator):
    def __init__(self, element):
        super().__init__(element)
    
    def _gen_full_instruction(self):
        return f"{randomInstruction()} the C++ class {self.element_name}"
    
    def _gen_first_fragment(self):
        return f"{randomInstruction()} the first fragment of the C++ class {self.element_name}"

    def _gen_next_fragment(self, idx):
        return f"{randomInstruction()} fragment (#{idx}) of the C++ class {self.element_name}"

    def _gen_last_fragment(self):
        return f"{randomInstruction()} the last fragment of the C++ class {self.element_name}"
        
    def _gen_docstring(self):
        return f"Write documentation (docstring) for the C++ class {self.element_name}"

class ConstructorTransformer(CppInstructionGenerator):
    def __init__(self, element):
        super().__init__(element)
    
    def _gen_full_instruction(self):
        return f"{randomInstruction()} the constructor of the C++ class {self.parent_name}"
    
    def _gen_first_fragment(self):
        return f"{randomInstruction()} the first fragment of the constructor of the C++ class {self.parent_name}"

    def _gen_next_fragment(self, idx):
        return f"{randomInstruction()} the next fragment (#{idx}) of the constructor of the C++ class {self.parent_name}"

    def _gen_last_fragment(self):
        return f"{randomInstruction()} the last fragment of the constructor of the C++ class {self.parent_name}"
        
    def _gen_docstring(self):
        return f"Write documentation (docstring) for the constructor of the C++ class {self.parent_name}"

class DestructorTransformer(CppInstructionGenerator):
    def __init__(self, element):
        super().__init__(element)
    
    def _gen_full_instruction(self):
        return f"{randomInstruction()} the destructor of the C++ class {self.parent_name}"
    
    def _gen_first_fragment(self):
        return f"{randomInstruction()} the first fragment of the destructor of the C++ class {self.parent_name}"

    def _gen_next_fragment(self, idx):
        return f"{randomInstruction()} the next fragment (#{idx}) of the destructor of the C++ class {self.parent_name}"

    def _gen_last_fragment(self):
        return f"{randomInstruction()} the last fragment of the destructor of the C++ class {self.parent_name}"
        
    def _gen_docstring(self):
        return f"Write documentation (docstring) for the destructor of the C++ class {self.parent_name}"

class CxxMethodTransformer(CppInstructionGenerator):
    def __init__(self, element):
        super().__init__(element)
    
    def _gen_full_instruction(self):
        return f"{randomInstruction()} the method {self.element_name} of the C++ class {self.parent_name}"
    
    def _gen_first_fragment(self):
        return f"{randomInstruction()} the first fragment of the method {self.element_name} of the C++ class {self.parent_name}"

    def _gen_next_fragment(self, idx):
        return f"{randomInstruction()} the next fragment (#{idx}) of the method {self.element_name} of the C++ class {self.parent_name}"

    def _gen_last_fragment(self):
        return f"{randomInstruction()} the last fragment of the method {self.element_name} of the C++ class {self.parent_name}"
        
    def _gen_docstring(self):
        return f"Write documentation (docstring) for the method {self.element_name} of the C++ class {self.parent_name}"

class DefaultDeclarationTransformer(CppInstructionGenerator):
    def __init__(self, element, type_name = None, suffix = None):
        super().__init__(element)
        self._type_name = type_name if type_name else "C++ element"
        self._suffic = f" of the {suffix}" if suffix else ""
    
    def _gen_full_instruction(self):
        
        return f"Declare the {self._type_name} {self.element_name}{self._suffic}"
    
    def _gen_first_fragment(self):
        return f"Declare the first fragment of the {self._type_name} {self.element_name}{self._suffic}"

    def _gen_next_fragment(self, idx):
        return f"Declare the next fragment (#{idx}) of the {self._type_name} {self.element_name}{self._suffic}"

    def _gen_last_fragment(self):
        return f"Declare the last fragment of the {self._type_name} {self.element_name}{self._suffic}"
        
    def _gen_docstring(self):
        return f"Write documentation (docstring) for the {self._type_name} {self.element_name}{self._suffic}"

def create_instruction_generator(element):
    if element["type"] == "function_decl":
        return FunctionTransformer(element)
    elif element["type"] == "cxx_method":
        return CxxMethodTransformer(element)
    elif element["type"] == "constructor":
        return ConstructorTransformer(element)
    elif element["type"] == "destructor":
        return DestructorTransformer(element)
    elif element["type"] == "function_template":
        if element["parent_type"] in ("class_decl", "struct_decl", "class_template", "class_template_partial_specialization"):
            return CxxMethodTransformer(element)
        else:
            return FunctionTransformer(element)
    elif element["type"] in ("class_decl", "struct_decl", "class_template", "class_template_partial_specialization"):
        return ClassTransformer(element)
    elif element["type"] == "var_decl":
        return DefaultDeclarationTransformer(element, type_name = "variable", suffix = "Tango Control sources")
    elif element["type"] == "enum_decl":
        return DefaultDeclarationTransformer(element, type_name = "enumeration", suffix = "Tango Control sources")
    else:
        raise NotImplementedError(f"Instruction generator for {element['type']} not implemented!")
    

def generate_instruction(element, mode: GenerationMode, idx=0):
    gen = create_instruction_generator(element)
    return gen.generate(mode, idx)


if __name__ == "__main__":
    element = {
            "type": "function_decl",
            "name": "calculateSum",
            "doc": "docstring",
            "parent_name": ""
        }
    
    modes = [GenerationMode.FULL, GenerationMode.FIRST_FRAGMENT, GenerationMode.NEXT_FRAGMENT, GenerationMode.LAST_FRAGMENT, GenerationMode.DOCSTRING]
    for md in modes:
        print(generate_instruction(element, md, 1))
