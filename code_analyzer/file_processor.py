import os
from pathlib import Path
from typing import List, Optional, Callable, Set
from clang.cindex import Index, TranslationUnit, Config
from interfaces import IFileProcessor

class FileProcessor(IFileProcessor):
    """Обработчик файлов и директорий проекта."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self._cached_include_dirs: Optional[List[str]] = None

    def find_source_files(self, extensions: Set[str] = None) -> List[Path]:
        """Найти все исходные файлы проекта."""
        extensions = extensions or {'.cpp', '.h', '.hpp', '.cc', '.cxx'}
        source_files = []
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if Path(file).suffix in extensions:
                    source_files.append(Path(root) / file)
        return source_files

    def get_include_dirs(self) -> List[str]:
        """Получить include-директории (с кешированием)."""
        if self._cached_include_dirs is None:
            self._cached_include_dirs = self._scan_include_dirs()
        return self._cached_include_dirs

    def _scan_include_dirs(self) -> List[str]:
        """Просканировать проект на наличие include-директорий."""
        include_dirs = [str(self.repo_path)]
        for root, dirs, _ in os.walk(self.repo_path):
            if 'include' in dirs:
                include_dirs.append(str(Path(root) / 'include'))
        return include_dirs

    def parse_file(self, file_path: Path, 
                  skip_if: Optional[Callable[[str], bool]] = None) -> Optional[TranslationUnit]:
        """Распарсить файл и вернуть TranslationUnit."""
        if skip_if and skip_if(file_path.name):
            return None
            
        args = self._get_compiler_args()
        try:
            return self.index.parse(str(file_path), args=args)
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
            return None

    def _get_compiler_args(self) -> List[str]:
        """Сформировать аргументы для компилятора."""
        args = [
            '-std=c++17',
            '-x', 'c++',
            '-fparse-all-comments',
            '-D__clang__'
        ]
        args.extend(f'-I{include}' for include in self.get_include_dirs())
        return args

    @staticmethod 
    def is_system_header(file_path: Optional[str]) -> bool:
        """Определить, является ли файл системным заголовком."""
        if not file_path:
            return True
        system_paths = {
            '/usr/include',
            '/usr/local/include',
            '/Applications/Xcode.app/',
            'C:\\Program Files (x86)\\Microsoft Visual Studio',
            'C:\\Program Files\\Microsoft Visual Studio',
            '<'            
        }
        return any(file_path.startswith(path) for path in system_paths) or ('\\Windows Kits\\' in file_path)