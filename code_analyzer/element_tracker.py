from typing import Set, Dict
from clang.cindex import Cursor, CursorKind
from pathlib import Path
from interfaces import IElementTracker

class ElementTracker(IElementTracker):
    
    def __init__(self):
        self._processed_elements = set()
        self.unprocessed_expected: Dict[str, int] = {}
        self.unprocessed_unexpected: Dict[str, int] = {}
        self.expected_unprocessed = (
            CursorKind.DECL_REF_EXPR,
            CursorKind.UNEXPOSED_EXPR,
            CursorKind.CALL_EXPR,
            CursorKind.PARM_DECL,
            CursorKind.TYPE_REF,
            CursorKind.NAMESPACE_REF,
            CursorKind.CXX_ACCESS_SPEC_DECL,
            CursorKind.COMPOUND_STMT,
            CursorKind.STRING_LITERAL,
            CursorKind.TEMPLATE_TYPE_PARAMETER,
            CursorKind.DECL_STMT,
            CursorKind.VAR_DECL,
            CursorKind.TEMPLATE_REF
        )

    def generate_element_id(self, cursor: Cursor, element_type: str) -> str:
        """Generate unique ID for an element to detect duplicates."""
        location = cursor.location.file.name if cursor.location.file else "unknown"
        return f"{element_type}:{location}:{cursor.location.line}:{cursor.spelling}"

    def is_processed(self, element_id: str) -> bool:
        """Check if element was already processed."""
        return element_id in self._processed_elements

    def mark_processed(self, element_id: str) -> None:
        """Mark element as processed."""
        self._processed_elements.add(element_id)

    def track_unhandled_kind(self, cursor: Cursor) -> None:
        """Track unhandled cursor kinds."""
        kind_name = str(cursor.kind)
        if cursor.kind in self.expected_unprocessed:
            self.unprocessed_expected[kind_name] = self.unprocessed_expected.get(kind_name, 0) + 1
        else:
            self.unprocessed_unexpected[kind_name] = self.unprocessed_unexpected.get(kind_name, 0) + 1

    @property
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about unprocessed elements."""
        return {
            "expected": dict(self.unprocessed_expected),
            "unexpected": dict(self.unprocessed_unexpected)
        }