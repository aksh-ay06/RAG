from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional,ContextManager

from sqlalchemy.orm import Session

from src import db

class BaseDatabase(ABC):
    @abstractmethod
    def startup(self) -> None:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        pass


class BaseRepository(ABC):
    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[Any]:
        pass

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        pass

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        pass

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        pass