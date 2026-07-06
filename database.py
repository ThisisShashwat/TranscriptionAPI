from typing import Generator
from sqlmodel import SQLModel, create_engine, Session
import config

engine = create_engine(
    f"sqlite:///{config.settings.db_path}",
    connect_args={"check_same_thread": False}
)

def init_db():
    import models
    SQLModel.metadata.create_all(engine)

def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
