import uuid
import secrets
import shutil
import logging
import httpx
import threading
from contextlib import asynccontextmanager
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
from sqlmodel import Session, select
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import config
import database
import manager
from models import (
    User, Hotword, Job, ConfigUpdate, ConfigResponse,
    UserRegisterResponse, JobSubmitResponse, JobListResponse, JobDetailResponse
)

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Phase
    config.settings.temp_dir.mkdir(exist_ok=True)
    database.init_db()

    stop_event = threading.Event()
    t_queue = threading.Thread(target=manager.process_queue_loop, args=(stop_event,), daemon=True)
    t_idle = threading.Thread(target=manager.idle_shutdown_monitor_loop, args=(stop_event,), daemon=True)
    
    t_queue.start()
    t_idle.start()
    logger.info("Application startup complete. Manager threads launched.")

    yield # Serves API requests here

    # Shutdown Phase
    logger.info("Shutting down application...")
    stop_event.set()
    manager.cleanup_active_runtimes()
    t_queue.join(timeout=5)
    t_idle.join(timeout=5)
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="Secured Transcription API Service",
    description="Refactored minimalist multi-tenant speech transcription API.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS and GZip Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

# SlowAPI Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Too Many Requests. Rate limit exceeded: {exc.detail}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact the administrator."}
    )

# API Authentication Security Scheme
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_current_user(
    api_key: str = Security(api_key_scheme),
    db: Session = Depends(database.get_db)
) -> User:
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header missing.")
    user = db.exec(select(User).where(User.api_key == api_key)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    return user

# Helper key function for API Key rate limiting
def get_api_key_str(request: Request) -> str:
    return request.headers.get("X-API-Key", "unknown")

# Rate limit string helper based on configuration
key_rate_limit_str = f"{config.settings.rate_limit_requests}/{config.settings.rate_limit_window}s"


# Authentication & Registration

@app.post("/auth/register", status_code=201, response_model=UserRegisterResponse, tags=["auth"])
@limiter.limit("10/minute")
def register_user(request: Request, db: Session = Depends(database.get_db)):
    """
    Register a new user account on the API server.
    Generates and returns a secure API Key.
    """
    while True:
        account_id = f"user_{secrets.token_hex(4)}"
        existing = db.exec(select(User).where(User.account_id == account_id)).first()
        if not existing:
            break
        
    api_key = f"transcribe_key_{secrets.token_hex(20)}"
    user = User(account_id=account_id, api_key=api_key)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "account_id": user.account_id,
        "api_key": user.api_key,
        "message": "The api key will only be shown one time. Save it. Next step: GET /auth/link"
    }


@app.get("/auth/link", tags=["auth"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def get_auth_link(
    request: Request,
    client_id: str = Query(..., description="Google OAuth Client ID from GCloud Console"),
    client_secret: str = Query("", description="Google OAuth Client Secret (optional)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Open google's OAuth to allow gcolab perms
    """
    client_id = client_id.strip()
    client_secret = client_secret.strip()
    
    if len(client_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid Google Client ID format.")
        
    current_user.client_id = client_id
    current_user.client_secret = client_secret
    db.add(current_user)
    db.commit()
    
    # Construct redirect URL pointing back to this API callback route
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    scopes = [
        "profile",
        "email",
        "https://www.googleapis.com/auth/colaboratory",
        "https://www.googleapis.com/auth/drive.file"
    ]
    
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": current_user.account_id
    }
    
    import urllib.parse
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(
            content=f'<a href="{auth_url}" target="_blank" style="font-family: sans-serif; font-size: 16px; color: #1a73e8; text-decoration: none; font-weight: bold;">Click here to Authorize with Google</a>',
            status_code=200
        )
    return {"auth_url": auth_url}


@app.get("/auth/callback", tags=["auth"], response_class=HTMLResponse)
@limiter.limit("10/minute")
async def auth_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(database.get_db)
):
    """
    Google OAuth redirect callback url
    """
    if error:
        logger.error(f"Google OAuth reported error: {error}")
        return HTMLResponse(f"<h3>Authentication failed: {error}</h3>", status_code=400)
        
    if not code or not state:
        return HTMLResponse("<h3>Authentication failed: Missing code or state parameters.</h3>", status_code=400)
        
    user = db.exec(select(User).where(User.account_id == state)).first()
    if not user:
        return HTMLResponse("<h3>Authentication failed: Invalid state metadata.</h3>", status_code=400)
        
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": user.client_id,
        "client_secret": user.client_secret or "",
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(token_url, data=payload)
            if r.status_code != 200:
                logger.error(f"Google Token Exchange returned {r.status_code}: {r.text}")
                return HTMLResponse(f"<h3>Authentication failed: Google token exchange returned status {r.status_code}</h3>", status_code=400)
            tokens = r.json()
            
        # Save tokens securely using Colab accounts store
        from sdk.colab_sdk import ColabAccountStore
        store = ColabAccountStore(config.settings.colab_accounts_file)
        store.add_profile(
            name=user.account_id,
            client_id=user.client_id,
            client_secret=user.client_secret or "",
            refresh_token=tokens.get("refresh_token", ""),
            access_token=tokens.get("access_token", ""),
            email=""
        )
        
        # Link user's profile inside database
        user.colab_profile_name = user.account_id
        db.add(user)
        db.commit()
        
        return """
        <html>
            <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 50px;">
                <h2 style="color: #2e7d32;">Authentication Successful</h2>
                <p>Your Google Colab account has been successfully linked to your API profile.</p>
                <p>You can close this tab now xD.</p>
            </body>
        </html>
        """
    except Exception as e:
        logger.error(f"Failed to exchange Google OAuth code: {e}")
        return HTMLResponse(f"<h3>Authentication failed while exchanging token: {e}</h3>", status_code=500)


@app.get("/auth/status", tags=["auth"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def get_auth_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Get connection status and stats of the user's Colab integration.
    """
    profile_name = current_user.colab_profile_name
    if not profile_name:
        return {"colab_linked": False, "message": "No Google Colab account linked to this profile."}
        
    try:
        from sdk.colab_sdk import ColabAccountStore
        store = ColabAccountStore(config.settings.colab_accounts_file)
        client = store.client(profile_name, verbose=False)
        auth = client.auth_status()
        runtimes = client.list_runtimes()
        
        return {
            "colab_linked": True,
            "email": auth.get("email", ""),
            "auth_ok": auth.get("ok", False),
            "token_seconds_remaining": auth.get("seconds_remaining", 0),
            "active_runtimes_count": len(runtimes)
        }
    except Exception as e:
        return {
            "colab_linked": True,
            "auth_ok": False,
            "error": str(e)
        }


#User Configurations

@app.get("/config", response_model=ConfigResponse, tags=["config"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def get_configurations(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve your global webhook URL and Gemini API Key config.
    """
    return {
        "webhook_url": current_user.webhook_url,
        "gemini_api_key": current_user.gemini_api_key
    }


@app.post("/config", tags=["config"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def update_configurations(
    request: Request,
    payload: ConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Update your global webhook URL and Gemini API Key.
    """
    if payload.webhook_url is not None:
        current_user.webhook_url = payload.webhook_url
    if payload.gemini_api_key is not None:
        current_user.gemini_api_key = payload.gemini_api_key
        
    db.add(current_user)
    db.commit()
    return {"status": "ok", "message": "User configurations updated successfully."}


# Custom Hotwords

@app.get("/hotwords", tags=["hotwords"], response_model=List[str])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def list_hotwords(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    List all custom hotwords registered under your account.
    """
    words = db.exec(select(Hotword.word).where(Hotword.user_id == current_user.id)).all()
    return words


@app.post("/hotwords", status_code=201, tags=["hotwords"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def create_hotword(
    request: Request,
    word: str = Query(..., min_length=2, max_length=64, description="Custom hotword to add"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Register a new custom hotword to correct spelling during transcription.
    """
    word_clean = word.strip()
    if not word_clean or len(word_clean) < 2:
        raise HTTPException(status_code=400, detail="Word must be at least 2 characters long.")
    if not all(x.isalnum() or x.isspace() or x == "'" for x in word_clean):
        raise HTTPException(status_code=400, detail="Word contains invalid characters.")
        
    hotword = Hotword(user_id=current_user.id, word=word_clean)
    try:
        db.add(hotword)
        db.commit()
    except Exception:
        db.rollback()
        # Silently ignore duplicate mapping constraint
        
    return {"status": "created", "word": word_clean}


@app.delete("/hotwords/{word}", tags=["hotwords"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def delete_hotword(
    request: Request,
    word: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Remove a hotword from your account.
    """
    hotword = db.exec(
        select(Hotword).where(Hotword.user_id == current_user.id, Hotword.word == word)
    ).first()
    if hotword:
        db.delete(hotword)
        db.commit()
    return {"status": "deleted", "word": word}


# Transcription & Queueing

@app.post("/transcribe", status_code=202, response_model=JobSubmitResponse, tags=["transcribe"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
async def enqueue_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    webhook_url: Optional[str] = Form(None, description="Optional. Webhook override for this request. Defaults to user's global webhook URL if not provided.", json_schema_extra={"example": "https://yourdomain.com/webhook"}),
    gemini_prompt: Optional[str] = Form(None, description="Optional. LLM prompt instructions for summary. If not provided and Gemini API key is configured, it defaults to summarizing the key points.", json_schema_extra={"example": "Summarize the key points of this transcription."}),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Queue an audio file for background transcription.
    Enforces file size limits and returns a job UUID immediately.
    """
    # 1. Input Validation: File size abuse prevention
    file.file.seek(0, 2)
    file_size_bytes = file.file.tell()
    file.file.seek(0)
    
    max_bytes = config.settings.max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Payload Too Large. Audio file size ({file_size_bytes / (1024*1024):.1f}MB) exceeds limit of {config.settings.max_file_size_mb}MB."
        )

    # 2. Check if Colab account is linked
    if not current_user.colab_profile_name:
        raise HTTPException(
            status_code=400,
            detail="Google Colab account must be linked before queueing transcription tasks."
        )
        
    # 3. Input Validation: Webhook URL override
    if webhook_url:
        webhook_clean = webhook_url.strip()
        if not (webhook_clean.startswith("http://") or webhook_clean.startswith("https://")):
            raise HTTPException(status_code=400, detail="Webhook URL must start with 'http://' or 'https://'")
            
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix or ".audio"
    temp_path = config.settings.temp_dir / f"{job_id}{suffix}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to write uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded audio file to API disk.")
        
    job = Job(
        job_id=job_id,
        user_id=current_user.id,
        status="pending",
        filename=file.filename,
        file_path=str(temp_path),
        webhook_url=webhook_url,
        gemini_prompt=gemini_prompt
    )
    db.add(job)
    db.commit()
    
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs", response_model=JobListResponse, tags=["transcribe"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def list_jobs(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Number of results to return (Pagination)"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    List transcription jobs submitted by your account (supports offset-based pagination).
    """
    jobs = db.exec(
        select(Job).where(Job.user_id == current_user.id)
        .order_by(Job.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    
    return {
        "limit": limit,
        "offset": offset,
        "count": len(jobs),
        "results": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "filename": j.filename,
                "created_at": j.created_at,
                "finished_at": j.finished_at,
                "retry_count": j.retry_count,
                "error": j.error_message
            }
            for j in jobs
        ]
    }


@app.get("/jobs/{job_id}", tags=["transcribe"])
@limiter.limit(key_rate_limit_str, key_func=get_api_key_str)
def get_job_status(
    request: Request,
    job_id: str,
    important_only: bool = Query(False, description="If true, returns only the essential fields: job_id, status, filename, created_at, finished_at, plain-text transcript, and the LLM summary."),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Retrieve details (status, transcript, LLM summary, errors) of a specific job.
    Supports an optional 'important_only' filter to return a simplified response.
    """
    job = db.exec(select(Job).where(Job.job_id == job_id)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied. You do not own this job.")
        
    if important_only:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "filename": job.filename,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "transcript": job.transcript.get("full_text") if job.transcript else None,
            "llm_summary": job.llm_summary
        }
        
    return {
        "job_id": job.job_id,
        "status": job.status,
        "filename": job.filename,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "transcript": job.transcript,
        "full_text": job.transcript.get("full_text") if job.transcript else None,
        "llm_summary": job.llm_summary,
        "retry_count": job.retry_count,
        "error": job.error_message
    }
