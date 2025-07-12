from abc import ABC, abstractmethod
from typing import Set, Dict, List, Optional, Callable
from pathlib import Path
from clang.cindex import Cursor, CursorKind, TranslationUnit


class IElementTracker(ABC):
    """Абстрактный интерфейс для отслеживания состояния обработки элементов кода.
    
    Предоставляет методы для:
    - Генерации уникальных идентификаторов элементов
    - Проверки и отметки обработанных элементов
    - Сбора статистики по необработанным элементам
    """
    
    @abstractmethod
    def is_processed(self, element_id: str) -> bool:
        """Проверить, был ли элемент обработан ранее.
        
        Args:
            element_id: Идентификатор элемента
            
        Returns:
            True если элемент уже был обработан, иначе False
        """
        pass
    
    @abstractmethod
    def mark_processed(self, element_id: str) -> None:
        """Пометить элемент как обработанный.
        
        Args:
            element_id: Идентификатор элемента
        """
        pass
    
    @abstractmethod
    def generate_element_id(self, cursor: Cursor) -> str:
        """Зарегистрировать необработанный вид курсора для статистики.
        
        Args:
            cursor: Курсор, который не был обработан
        """
        pass
    
    @property
    @abstractmethod
    def unprocessed_stats(self) -> Dict[str, Dict[str, int]]:
        """Статистика по необработанным курсорам.
        
        Returns:
            Словарь с двумя подразделами:
            - 'expected': {kind_name: count} - ожидаемые необработанные виды
            - 'unexpected': {kind_name: count} - неожиданные необработанные виды
        """
        pass


class IFileProcessor(ABC):
    """Абстрактный интерфейс для работы с файловой системой и парсинга исходников.
    
    Предоставляет методы для:
    - Поиска исходных файлов в проекте
    - Парсинга файлов через libclang
    - Работы с include-директориями
    """

    @abstractmethod
    def find_source_files(self, extensions: Set[str]) -> List[Path]:
        """Найти все исходные файлы проекта с указанными расширениями.
        
        Args:
            extensions: Множество расширений файлов (например, {'.cpp', '.hpp'})
            
        Returns:
            Список абсолютных путей к файлам
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path, 
                  skip_if: Optional[Callable[[str], bool]] = None) -> Optional[TranslationUnit]:
        """Распарсить исходный файл и получить AST.
        
        Args:
            file_path: Путь к файлу для парсинга
            skip_if: Опциональная функция-предикат для пропуска файлов
            
        Returns:
            TranslationUnit libclang или None если файл пропущен/не удалось распарсить
        """
        pass
    
    @abstractmethod
    def is_system_header(self, file_path: Optional[str]) -> bool:
        """Проверить, является ли файл системным заголовком.
        
        Args:
            file_path: Путь к файлу или None
            
        Returns:
            True если файл является системным заголовком, иначе False
        """
        pass

    @abstractmethod
    def get_relative_path(self, absolute_path: str) -> str:
        pass

    @property
    @abstractmethod
    def include_dirs(self) -> List[str]:
        """Список include-директорий проекта.
        
        Returns:
            Список абсолютных путей к директориям с заголовками
        """
        pass