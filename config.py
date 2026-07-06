from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    port: int = 8000
    db_path: str = str(BASE_DIR / "transcription.db")

    colab_accounts_file: str = str(BASE_DIR / "colab_accounts.json")

    idle_timeout_sec: int = 300

    temp_dir: Path = BASE_DIR / "_temp_uploads"

    rate_limit_requests: int = 60
    rate_limit_window: int = 60
    max_file_size_mb: int = 100

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTION_",
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
