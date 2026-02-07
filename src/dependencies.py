from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request

from sqlalchemy.orm import Session
from src.config import Settings
from src.db.interfaces.base import BaseDatabase



@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    with database.get_session() as session:
        yield session


def get_pdf_parser_service(request: Request):
    return None


def get_opensearch_service(request: Request):
    return getattr(request.app.state, "opensearch_service", None)



def get_llm_service(request: Request):
    return None


# Dependency type aliases for better type hints
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
PDFParserServiceDep = Annotated[object, Depends(get_pdf_parser_service)]
OpenSearchServiceDep = Annotated[object, Depends(get_opensearch_service)]
