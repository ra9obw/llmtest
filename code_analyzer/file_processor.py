import os
from pathlib import Path
from typing import List, Optional, Callable, Set
from clang.cindex import Index, TranslationUnit, Config
from interfaces import IFileProcessor

class FileProcessor(IFileProcessor):
    """Default implementation of IFileProcessor for CodeExtractor."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()
        self.index = Index.create()
        self._include_dirs = self._find_include_dirs()

    def _find_include_dirs(self) -> List[str]:
        """Find all include directories in the repository."""
        include_dirs = [str(self.repo_path)]
        for root, dirs, _ in os.walk(self.repo_path):
            if 'include' in dirs:
                include_dirs.append(str(Path(root) / 'include'))
        return include_dirs

    def find_source_files(self, extensions: Set[str]) -> List[Path]:
        """Find all source files with given extensions in the repository."""
        source_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    source_files.append(Path(root) / file)
        return source_files

    def parse_file(self, file_path: Path, 
                  skip_if: Optional[Callable[[str], bool]] = None) -> Optional[TranslationUnit]:
        """Parse a source file and return its AST."""
        if skip_if and skip_if(str(file_path)):
            return None
            
        args = [
            '-std=c++17',
            '-x', 'c++',
            '-fparse-all-comments',
            '-D__clang__'
        ]
        args.extend(arg for include_dir in self.include_dirs 
                   for arg in ['-I', include_dir])
        
        return self.index.parse(str(file_path), args=args)

    def is_system_header(self, file_path: Optional[str]) -> bool:
        """Check if the file is a system header."""
        if not file_path:
            return True
        return file_path.startswith('/usr/include') or \
               file_path.startswith('/usr/local/include') or \
               file_path.startswith('/Applications/Xcode.app/') or \
               file_path.startswith('C:\\Program Files (x86)\\Microsoft Visual Studio') or \
               file_path.startswith('C:\\Program Files\\Microsoft Visual Studio') or \
               '\\Windows Kits\\' in file_path or \
               file_path.startswith('<')
    
    def get_relative_path(self, absolute_path: str) -> str:
        try:
            return str(Path(absolute_path).resolve().relative_to(self.repo_path))
        except ValueError:
            return absolute_path
    
    @property
    def include_dirs(self) -> List[str]:
        """Get list of include directories."""
        return self._include_dirs
