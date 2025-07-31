from abc import ABC, abstractmethod
from typing import Set, Dict, List, Optional, Callable, Any
from pathlib import Path
from clang.cindex import Cursor, CursorKind, TranslationUnit


class IElementTracker(ABC):
    
    @abstractmethod
    def is_processed(self, element_id: str) -> bool:
        pass
    
    @abstractmethod
    def mark_processed(self, element_id: str) -> None:
        pass
    
    @abstractmethod
    def generate_name(self, cursor: Cursor) -> str:
        pass

    @abstractmethod
    def generate_element_id(self, cursor: Cursor) -> str:
        pass

    @abstractmethod
    def track_unhandled_kind(self, cursor: Cursor) -> None:
        pass

    @property
    @abstractmethod
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        pass


class IFileProcessor(ABC):

    @abstractmethod
    def find_source_files(self, extensions: Set[str]) -> List[Path]:
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path, 
                  skip_if: Optional[Callable[[str], bool]] = None) -> Optional[TranslationUnit]:
        pass
    
    @abstractmethod
    def is_system_header(self, file_path: Optional[str]) -> bool:
        pass

    @abstractmethod
    def get_relative_path(self, absolute_path: str) -> str:
        pass

    @property
    @abstractmethod
    def include_dirs(self) -> List[str]:
        pass

class ICodeCleaner(ABC):
    @staticmethod
    @abstractmethod
    def clean_code(code: str) -> Optional[str]:
        pass


class IRangeLocator(ABC):
    @abstractmethod
    def get_sibling_and_parent_positions(self, cursor: Cursor) -> List[Dict]:
        pass

    @abstractmethod
    def get_next_cursor_position(self, cursor: Cursor) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_previous_cursor_position(self, cursor: Cursor) -> Optional[Dict]:
        pass

    @abstractmethod
    def get_code_snippet(self, cursor: Cursor) -> Optional[str]:
        pass

    @abstractmethod
    def get_context(self, cursor) -> Dict[str, List[str]]:
        pass

    @abstractmethod
    def get_comments_in_range(self, file_path, start_line, end_line):
        pass

    @staticmethod
    @abstractmethod
    def get_offset_from_position(file_path: str, line: int, column: int) -> int:
        pass

    @staticmethod
    @abstractmethod
    def get_offset(location) -> int:
        pass


class IJsonDataStorage(ABC):
    @abstractmethod
    def add_element(self, element_type: str, element_data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def create_element_storage(self, fields: Set[str]) -> None:
        pass

    @abstractmethod
    def save_to_file(self) -> None:
        pass

    @abstractmethod
    def print_statistics(self, unprocessed_stats: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        pass
