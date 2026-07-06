from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON, UniqueConstraint
from pydantic import BaseModel, field_validator

# db tables

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    account_id: str = Field(unique=True, index=True)
    api_key: str = Field(unique=True, index=True)
    colab_profile_name: Optional[str] = Field(default=None, unique=True)
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    webhook_url: Optional[str] = None
    gemini_api_key: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # ORM
    hotwords: List["Hotword"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    jobs: List["Job"] = Relationship(back_populates="user")


class Hotword(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("user_id", "word", name="unique_user_word"),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", ondelete="CASCADE")
    word: str
    added_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    user: User = Relationship(back_populates="hotwords")


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(unique=True, index=True)
    user_id: int = Field(foreign_key="user.id")
    status: str = "pending"  # pending, starting_colab, uploading, processing, done, failed
    filename: str
    file_path: str
    webhook_url: Optional[str] = None
    gemini_prompt: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: Optional[str] = None

    transcript: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    llm_summary: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)

    user: User = Relationship(back_populates="jobs")


# api validatins


class ConfigUpdate(BaseModel):
    webhook_url: Optional[str] = Field(None, max_length=256, description="endpoint for completion webhooks")
    gemini_api_key: Optional[str] = Field(None, min_length=10, max_length=120, description="Google Gemini Studio API Key")

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_strip = v.strip()
            if v_strip:
                if not (v_strip.startswith("http://") or v_strip.startswith("https://")):
                    raise ValueError("Webhook URL must start with 'http://' or 'https://'")
                return v_strip
            return None
        return v


# api schemas

class ConfigResponse(BaseModel):
    webhook_url: Optional[str] = None
    gemini_api_key: Optional[str] = None


class UserRegisterResponse(BaseModel):
    account_id: str
    api_key: str
    message: str


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobSummary(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: str
    finished_at: Optional[str]
    retry_count: int
    error: Optional[str] = None


class JobListResponse(BaseModel):
    limit: int
    offset: int
    count: int
    results: List[JobSummary]


class JobDetailResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    created_at: str
    finished_at: Optional[str]
    transcript: Optional[Dict[str, Any]]
    full_text: Optional[str] = None
    llm_summary: Optional[str]
    retry_count: int
    error: Optional[str] = None
