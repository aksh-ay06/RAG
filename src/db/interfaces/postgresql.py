import logging 
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Type


from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine,inspect,text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.db.interfaces.base import BaseDataBase

logger = logging.getLogger(__name__)

class PostgreSQLsettings(BaseSettings):
    # database_url: str = Field(..., env="DATABASE_URL", description="Database connection URL")
    database_url: str = Field(default="postgresql://rag_user:rag_password@localhost:5432/rag_db", description="PostgreSQL database URL")
    echo_sql: bool = Field(False, description="Enable SQLAlchemy echo for debugging")
    pool_size: int = Field(20, description="Database connection pool size")
    max_overflow: int = Field(0, description="Maximum overflow size for the connection pool")

    class Config:
        # env_file = ".env"
        # env_file_encoding = "utf-8"
        env_prefix = "POSTGRESQL_"


Base = declarative_base()

class PostgreSQLDatabase(BaseDataBase):
    def __init__(self, config: PostgreSQLsettings):
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None


    def startup(self) -> None:
        try:
            logger.info("Starting up PostgreSQL database connection...")
            self.engine = create_engine(
                self.config.database_url,
                echo=self.config.echo_sql,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
            )
            self.sesion_factory = sessionmaker(bind=self.engine,expire_on_commit=False)

            assert self.engine is not None
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("PostgreSQL database connection established successfully.")

            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            logger.info(f"Existing tables in the database: {existing_tables}")

            Base.metadata.create_all(self.engine)

            updated_tables = inspector.get_table_names()
            new_tables = set(updated_tables) - set(existing_tables)
            
            if new_tables:
                logger.info(f"New tables created: {new_tables}")
            else:
                logger.info("No new tables were created. All tables already exist.")
            
            logger.info("PostgreSQL database startup completed successfully.")
            assert self.engine is not None
            logger.info(f"Database URL: {self.engine.url.database}")
            logger.info(f"Total tables: {','.join(updated_tables) if updated_tables else 'None'}")
            logger.info("Database connection established")
        
        except Exception as e:
            logger.error(f"Error during PostgreSQL database startup: {e}")
            raise
    
    def teardown(self) -> None:
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL database connection closed.")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        if not self.sesion_factory:
            raise RuntimeError("Session factory is not initialized. Call startup() first.")
        session = self.sesion_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error during database session: {e}")
            raise
        finally:
            session.close()