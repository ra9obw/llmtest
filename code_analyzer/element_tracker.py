from typing import Set, Dict
from clang.cindex import Cursor, CursorKind
from pathlib import Path
from interfaces import IElementTracker

class ElementTracker(IElementTracker):
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self._processed: Set[str] = set()
        self._unprocessed_expected: Dict[str, int] = {}
        self._unprocessed_unexpected: Dict[str, int] = {}
    
    def generate_element_id(self, cursor: Cursor, element_type: str) -> str:
        rel_path = self._get_relative_path(cursor.location.file.name)
        return f"{element_type}:{rel_path}:{cursor.location.line}:{cursor.hash}"

    def is_processed(self, element_id: str) -> bool:
        return element_id in self._processed

    def mark_processed(self, element_id: str) -> None:
        self._processed.add(element_id)

    def track_unhandled_kind(self, cursor: Cursor) -> None:
        kind_name = str(cursor.kind)
        if cursor.kind in self.expected_kinds:
            self._unprocessed_expected[kind_name] = self._unprocessed_expected.get(kind_name, 0) + 1
        else:
            self._unprocessed_unexpected[kind_name] = self._unprocessed_unexpected.get(kind_name, 0) + 1

    @property
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        return {
            'expected': self._unprocessed_expected,
            'unexpected': self._unprocessed_unexpected
        }

    def _get_relative_path(self, absolute_path: str) -> str:
        try:
            return str(Path(absolute_path).relative_to(self.repo_path))
        except ValueError:
            return absolute_path