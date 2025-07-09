import re
from typing import Optional

class CodeCleaner:
    """Класс для очистки и нормализации кода C++."""
    
    @staticmethod
    def clean_code(code: str) -> Optional[str]:
        """Clean code while preserving meaningful indentation."""
        if not code:
            return None

        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        
        while lines and not lines[0].strip():
            lines.pop(0)
        
        while lines and not lines[-1].strip():
            lines.pop()
        
        lines = [line.replace('\t', '    ') for line in lines]
        code = '\n'.join(lines)
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        return code