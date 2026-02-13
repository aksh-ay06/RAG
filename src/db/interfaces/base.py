from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional,ContextManager

from sqlalchemy.orm import Session

from src import db

class BaseDatabase(ABC):
    @abstractmethod
    def startup(self) -> None:
        """Initialize the database connection."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Get a new database session."""


class BaseRepository(ABC):

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record in the database."""

    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[Any]:
        """Retrieve a record by its ID."""

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update an existing record in the database."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record from the database."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records in the database with pagination."""