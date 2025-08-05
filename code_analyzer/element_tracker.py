from typing import Dict, Optional, Callable
from clang.cindex import Cursor, CursorKind
from pathlib import Path
from interfaces import IElementTracker
import hashlib

class ElementTracker(IElementTracker):
    
    def __init__(self, get_relative_path: Optional[Callable[[str], str]] = None):
        self._processed_elements = set()
        self.unprocessed_expected: Dict[str, int] = {}
        self.unprocessed_unexpected: Dict[str, int] = {}
        self.get_relative_path = get_relative_path
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
            CursorKind.TEMPLATE_REF,
            CursorKind.CXX_BASE_SPECIFIER,
            CursorKind.FRIEND_DECL,
            CursorKind.INTEGER_LITERAL,
            CursorKind.CXX_BOOL_LITERAL_EXPR
        )

    def generate_name(self, cursor: Cursor) -> str:
        # print(f"{cursor.kind}\t{cursor.spelling}")
        if cursor.spelling and not cursor.spelling.startswith("(unnamed struct") and not cursor.spelling.startswith("(anonymous union"):
            _name = cursor.spelling
        else:
            _kind = cursor.kind.name
            if self.get_relative_path:
                _file_path = self.get_relative_path(cursor.location.file.name)
            else:
                _file_path = cursor.location.file.name
            _name = f"{_kind[:5]}-{_file_path}-{cursor.location.line}-{cursor.location.column}"
        return _name

    def generate_element_id(self, cursor: Cursor) -> str:
        """Generates a unique ID for a cursor based on its metadata.

        Format: {type_prefix}_{12_char_hash}
        The hash is computed from: cursor type, file path, position, and name.
        """
        _kind = cursor.kind.name
        _name = self.generate_name(cursor)
        if cursor.spelling:
            if not cursor.location.file:
                _unique_str = f"{_kind}:{_name}"  # elements with no file
            else:
                if self.get_relative_path:
                    _file_path = self.get_relative_path(cursor.location.file.name).replace(":", "_")
                else:
                    _file_path = cursor.location.file.name.replace(":", "_")
                _unique_str = f"{_kind}-{_file_path}-{cursor.location.line}-{cursor.location.column}-{_name}"
        else:
            _unique_str = f"{_name}"
        # return _unique_str
        _hash_part = hashlib.sha256(_unique_str.encode()).hexdigest()[:12]  # первые 8 символов = 32 бита
        _id = f"{_kind[:5]}_{_hash_part}"  # например: "CLA_a1b2c3d4"
        return _id

    def generate_short_id(self, cursor: Cursor) -> str:
        return self.generate_element_id(cursor)
        # full_id = self.generate_element_id(cursor)
        # _hash_part = hashlib.sha256(full_id.encode()).hexdigest()[:12]  # первые 8 символов = 32 бита
        # _kind = cursor.kind.name
        # _id = f"{_kind[:5]}_{_hash_part}"  # например: "CLA_a1b2c3d4"
        # return _id

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