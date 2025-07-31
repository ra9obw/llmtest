class CppSignatureGenerator:
    def __init__(self, element):
        self.signature = element["signature"]
        self.parent_name = element.get("parent_name")
    
    def _get_qualifiers(self):
        qualifiers = []
        if self.signature["qualifiers"]["is_static"]:
            qualifiers.append("static")
        if self.signature["qualifiers"]["is_virtual"]:
            qualifiers.append("virtual")
        if self.signature["qualifiers"]["is_const"]:
            qualifiers.append("const")
        if self.signature["qualifiers"]["is_noexcept"]:
            qualifiers.append("noexcept")
        if self.signature["qualifiers"]["is_pure_virtual"]:
            qualifiers.append("= 0")
        return qualifiers
    
    def _get_parameters(self):
        parameters = []
        for param in self.signature["parameters"]:
            param_str = f"{param['type']}"
            if param["name"]:
                param_str += f" {param['name']}"
            if param["default_value"] is not None:
                param_str += f" = {param['default_value']}"
            parameters.append(param_str)
        return parameters
    
    def generate(self):
        raise NotImplementedError("Subclasses must implement this method")


class FunctionSignatureGenerator(CppSignatureGenerator):
    def generate(self):
        qualifiers = self._get_qualifiers()
        parameters = self._get_parameters()
        
        parts = []
        if qualifiers and not self.signature["qualifiers"]["is_pure_virtual"]:
            parts.extend(qualifiers[:-1])
        
        parts.append(self.signature["return_type"])
        parts.append(f"{self.signature['name']}({', '.join(parameters)})")
        
        if self.signature["qualifiers"]["is_const"] and "const" in qualifiers:
            parts.append("const")
        if self.signature["qualifiers"]["is_noexcept"] and "noexcept" in qualifiers:
            parts.append("noexcept")
        if self.signature["qualifiers"]["is_pure_virtual"] and "= 0" in qualifiers:
            parts.append("= 0")
        
        return " ".join(parts)


class MethodSignatureGenerator(CppSignatureGenerator):
    def generate(self):
        qualifiers = self._get_qualifiers()
        parameters = self._get_parameters()
        
        parts = []
        if qualifiers and not self.signature["qualifiers"]["is_pure_virtual"]:
            parts.extend(qualifiers[:-1])
        
        parts.append(self.signature["return_type"])
        parts.append(f"{self.parent_name}::{self.signature['name']}({', '.join(parameters)})")
        
        if self.signature["qualifiers"]["is_const"] and "const" in qualifiers:
            parts.append("const")
        if self.signature["qualifiers"]["is_noexcept"] and "noexcept" in qualifiers:
            parts.append("noexcept")
        if self.signature["qualifiers"]["is_pure_virtual"] and "= 0" in qualifiers:
            parts.append("= 0")
        
        return " ".join(parts)


class ConstructorSignatureGenerator(CppSignatureGenerator):
    def generate(self):
        qualifiers = self._get_qualifiers()
        parameters = self._get_parameters()
        
        parts = []
        if qualifiers and not self.signature["qualifiers"]["is_pure_virtual"]:
            parts.extend(qualifiers[:-1])
        
        parts.append(f"{self.parent_name}::{self.signature['name']}({', '.join(parameters)})")
        
        if self.signature["qualifiers"]["is_noexcept"] and "noexcept" in qualifiers:
            parts.append("noexcept")
        
        return " ".join(parts)


class DestructorSignatureGenerator(CppSignatureGenerator):
    def generate(self):
        qualifiers = self._get_qualifiers()
        
        parts = []
        if qualifiers and not self.signature["qualifiers"]["is_pure_virtual"]:
            parts.extend(qualifiers[:-1])
        
        parts.append(f"{self.parent_name}::~{self.signature['name']}()")
        
        if self.signature["qualifiers"]["is_noexcept"] and "noexcept" in qualifiers:
            parts.append("noexcept")
        if self.signature["qualifiers"]["is_pure_virtual"] and "= 0" in qualifiers:
            parts.append("= 0")
        
        return " ".join(parts)


def create_signature_generator(element):
    if element["type"] == "function_decl":
        return FunctionSignatureGenerator(element)
    elif element["type"] == "cxx_method":
        return MethodSignatureGenerator(element)
    elif element["type"] == "constructor":
        return ConstructorSignatureGenerator(element)
    elif element["type"] == "destructor":
        return DestructorSignatureGenerator(element)
    else:
        raise NotImplementedError(f"Signature generator for {element["type"]} not implemented!")
    

def generate_signature(element):
    gen = create_signature_generator(element)
    return gen.generate()