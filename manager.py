import os
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from google import genai
from sqlmodel import Session, select

import config
import database
from models import User, Job, Hotword
from sdk.colab_sdk import ColabAccountStore, ColabQuotaError, ColabBlockedError
from sdk.whisper_sdk import WhisperClient

logger = logging.getLogger("manager")
logger.setLevel(logging.INFO)

# user_id -> {"client": ColabClient, "runtime": ColabRuntime, "url": str, "last_active": float}
active_runtimes: Dict[int, Dict[str, Any]] = {}
active_runtimes_lock = threading.Lock()

# Read the whisper server script from the same directory
SERVER_SCRIPT_PATH = Path(__file__).resolve().parent / "sdk/whisper_server.py"
try:
    with open(SERVER_SCRIPT_PATH, "r", encoding="utf-8") as f:
        SERVER_SCRIPT = f.read()
except Exception as e:
    logger.error(f"Failed to read whisper_server.py: {e}")
    SERVER_SCRIPT = ""

def send_webhook(url: str, payload: dict):
    """Webhook post using a background thread."""
    def _fire():
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.post(url, json=payload)
                logger.info(f"Webhook delivered to {url}. Response code: {r.status_code}")
        except Exception as e:
            logger.error(f"Webhook failed to deliver to {url}: {e}")
    threading.Thread(target=_fire, daemon=True).start()

def call_gemini(api_key: str, prompt: str, transcript_text: str) -> Optional[str]:
    """Generates LLM summaries using gemma-4-31b-it."""
    try:
        client = genai.Client(api_key=api_key)
        full_prompt = f'Using this transcription:\n"""\n{transcript_text}\n"""\n\nPerform the following task: {prompt}'
        logger.info("Attempting LLM summary with gemma-4-31b-it...")
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemma LLM call failed: {e}")
        return None

def get_or_start_colab(user: User) -> tuple[Any, str]:
    """Orchestrates runtime allocation and dependencies installation on Colab."""
    user_id = user.id
    profile_name = user.colab_profile_name
    if not profile_name:
        raise ValueError("Colab account not linked.")

    with active_runtimes_lock:
        if user_id in active_runtimes:
            active_runtimes[user_id]["last_active"] = time.time()
            return active_runtimes[user_id]["runtime"], active_runtimes[user_id]["url"]

    store = ColabAccountStore(config.settings.colab_accounts_file)
    client = store.client(profile_name, verbose=False)

    # Clean any existing runtimes on the Colab account first to prevent 412 Precondition Failed
    for rt in client.list_runtimes():
        logger.info(f"[{user.account_id}] Found existing runtime '{rt.endpoint}'. Unassigning it first...")
        try:
            client.unassign_runtime(rt)
        except Exception as e:
            logger.warning(f"[{user.account_id}] Failed to unassign runtime: {e}")

    logger.info(f"[{user.account_id}] Provisioning Colab T4 GPU...")
    rt = None
    try:
        rt = client.runtime(variant="GPU", accelerator="T4", auto_unassign=False)
    except Exception as e:
        err_msg = str(e)
        if "412" in err_msg or "Precondition Failed" in err_msg:
            logger.warning(f"[{user.account_id}] Allocation failed with 412 Precondition Failed. Retrying unassign cleanup...")
            for active_rt in client.list_runtimes():
                try:
                    client.unassign_runtime(active_rt)
                except Exception:
                    pass
            # Wait a moment for Colab backend state to settle
            time.sleep(3.0)
            try:
                rt = client.runtime(variant="GPU", accelerator="T4", auto_unassign=False)
            except ColabQuotaError:
                raise RuntimeError("Google Colab quota exhausted on linked account.")
            except ColabBlockedError:
                raise RuntimeError("Google Colab account has been blocked/restricted.")
            except Exception as retry_err:
                raise RuntimeError(f"Runtime allocation retry failed: {retry_err}")
        else:
            if isinstance(e, ColabQuotaError):
                raise RuntimeError("Google Colab quota exhausted on linked account.")
            if isinstance(e, ColabBlockedError):
                raise RuntimeError("Google Colab account has been blocked/restricted.")
            raise RuntimeError(f"Runtime allocation failed: {e}")

    logger.info(f"[{user.account_id}] Runtime assigned. Deploying dependencies...")
    # Install dependencies
    pre_installs = [
        "wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared",
        "chmod +x cloudflared",
        'pip install -q faster-whisper fastapi "uvicorn[standard]" python-multipart'
    ]
    for cmd in pre_installs:
        rt.run_shell(cmd)

    # Clean up any existing server/tunnel processes on the runtime first (e.g. from previous aborted runs)
    try:
        logger.info(f"[{user.account_id}] Cleaning up any existing server/tunnel processes on Colab runtime...")
        rt.run_shell("fuser -k 8000/tcp || true")
        rt.run_shell("pkill -f cloudflared || true")
    except Exception as cleanup_err:
        logger.warning(f"[{user.account_id}] Process cleanup warning: {cleanup_err}")

    # Run whisper_server.py script
    logger.info(f"[{user.account_id}] Starting Whisper server script on Colab...")
    job = rt.start_job(SERVER_SCRIPT, name=f"whisper-server-{user.account_id}")
    watcher = job.watcher()
    url = watcher.wait_for_pattern(r"https://[^\s]+\.trycloudflare\.com", timeout=720)

    if not url:
        try:
            tail = job.tail(50)
            logger.error(f"[{user.account_id}] Whisper server failed to start. Last 50 lines of output:")
            for line in tail.splitlines():
                logger.error(f"  [Colab Output] {line}")
        except Exception as tail_err:
            logger.error(f"[{user.account_id}] Failed to fetch job output tail: {tail_err}")

        try:
            client.unassign_runtime(rt.assignment.endpoint)
        except Exception:
            pass
        raise TimeoutError("Cloudflare tunnel failed to start on Colab instance.")

    # Wait for the Cloudflare tunnel domain DNS propagation and server health responsiveness
    logger.info(f"[{user.account_id}] Waiting for Cloudflare tunnel URL '{url}' to become responsive (health check)...")
    import requests
    start_time = time.time()
    url_ok = False
    while time.time() - start_time < 240.0:  # Wait up to 240 seconds
        try:
            r = requests.get(f"{url}/health", timeout=3.0)
            if r.status_code == 200:
                url_ok = True
                break
        except Exception:
            pass
        time.sleep(3.0)

    if not url_ok:
        try:
            client.unassign_runtime(rt.assignment.endpoint)
        except Exception:
            pass
        raise TimeoutError(f"Cloudflare tunnel URL {url} did not become responsive (health check failed).")

    logger.info(f"[{user.account_id}] Whisper server ready at: {url}")
    
    # Start the Colab control plane keep-alive loop (ping every 5 mins)
    rt.start_keepalive(interval=300.0)

    with active_runtimes_lock:
        active_runtimes[user_id] = {
            "client": client,
            "runtime": rt,
            "url": url,
            "last_active": time.time()
        }

    return rt, url

def process_queue_loop(stop_event: threading.Event):
    """Processes jobs from the DB queue sequentially (FIFO order) with retries and exponential backoff."""
    logger.info("Starting background queue worker loop...")
    while not stop_event.is_set():
        with Session(database.engine) as session:
            job = session.exec(
                select(Job).where(Job.status == "pending").order_by(Job.id.asc()).limit(1)
            ).first()
            if not job:
                stop_event.wait(3.0)
                continue

            job_id = job.job_id
            user_id = job.user_id
            file_path = job.file_path

            # Load User details
            user = session.get(User, user_id)
            if not user:
                logger.error(f"Job {job_id} owner missing. Marking failed.")
                job.status = "failed"
                job.error_message = "Owner profile missing."
                job.finished_at = datetime.utcnow().isoformat()
                session.add(job)
                session.commit()
                continue

        max_retries = 5
        backoff_base = 10.0
        success = False
        error_msg = ""

        for attempt in range(1, max_retries + 1):
            try:
                # 1. Starting Colab runtime setup
                logger.info(f"Processing job {job_id} (Attempt {attempt}/{max_retries}) for user {user.account_id}...")
                with Session(database.engine) as session:
                    db_job = session.get(Job, job.id)
                    if db_job:
                        db_job.status = "starting_colab"
                        session.add(db_job)
                        session.commit()

                rt, server_url = get_or_start_colab(user)

                # 2. Set uploading status
                with Session(database.engine) as session:
                    db_job = session.get(Job, job.id)
                    if db_job:
                        db_job.status = "uploading"
                        session.add(db_job)
                        session.commit()

                if not os.path.exists(file_path):
                    raise FileNotFoundError("Audio file was cleaned up on local disk.")

                # Submit to Whisper Colab server
                client = WhisperClient(server_url=server_url, verbose=False, timeout=120, upload_timeout=600)
                
                with Session(database.engine) as session:
                    hotwords_list = session.exec(select(Hotword.word).where(Hotword.user_id == user_id)).all()
                hotwords_str = ",".join(hotwords_list) if hotwords_list else None

                colab_job_id = client.submit(audio_path=file_path, model="large-v3", task="transcribe", hotwords=hotwords_str)

                # 3. Transition to processing
                with Session(database.engine) as session:
                    db_job = session.get(Job, job.id)
                    if db_job:
                        db_job.status = "processing"
                        session.add(db_job)
                        session.commit()

                # Block and wait for transcription result
                result = client.wait(colab_job_id)
                if not result:
                    raise ValueError("Empty response from transcription client.")

                # 4. Gemini/Gemma Post-processing
                llm_summary = None
                if user.gemini_api_key:
                    prompt = job.gemini_prompt or "Summarize the key points of this transcription."
                    llm_summary = call_gemini(user.gemini_api_key, prompt, result.get("full_text", ""))

                # Successful finish
                with Session(database.engine) as session:
                    db_job = session.get(Job, job.id)
                    if db_job:
                        db_job.status = "done"
                        db_job.transcript = result
                        db_job.llm_summary = llm_summary
                        db_job.finished_at = datetime.utcnow().isoformat()
                        session.add(db_job)
                        session.commit()
                logger.info(f"Job {job_id} successfully completed.")
                success = True

                # Dispatch Success Webhook
                webhook_url = job.webhook_url or user.webhook_url
                if webhook_url:
                    send_webhook(webhook_url, {
                        "job_id": job_id,
                        "status": "done",
                        "filename": job.filename,
                        "transcript": result,
                        "llm_summary": llm_summary,
                        "error": None
                    })
                break  # Successful execution, break the retry loop

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.warning(f"Job {job_id} (Attempt {attempt}/{max_retries}) failed: {error_msg}")
                
                # Force close and clear the active Colab runtime session for this user
                # to guarantee that the next retry attempt provisions a clean runtime VM.
                with active_runtimes_lock:
                    if user_id in active_runtimes:
                        info = active_runtimes[user_id]
                        try:
                            info["runtime"].close()
                            info["client"].unassign_runtime(info["runtime"].assignment.endpoint)
                        except Exception as unassign_err:
                            logger.debug(f"Failed to close and unassign runtime during retry cleanup: {unassign_err}")
                        del active_runtimes[user_id]
                
                # Update retry count and last error in the database
                with Session(database.engine) as session:
                    db_job = session.get(Job, job.id)
                    if db_job:
                        db_job.retry_count = attempt
                        db_job.error_message = f"[Attempt {attempt}/{max_retries} failed] {error_msg}"
                        session.add(db_job)
                        session.commit()
                
                if attempt < max_retries:
                    sleep_time = backoff_base * (2 ** (attempt - 1))
                    logger.info(f"Retrying job {job_id} in {sleep_time}s...")
                    stop_event.wait(sleep_time)

        if not success:
            logger.error(f"Job {job_id} failed permanently after {max_retries} attempts.")
            with Session(database.engine) as session:
                db_job = session.get(Job, job.id)
                if db_job:
                    db_job.status = "failed"
                    db_job.error_message = error_msg
                    db_job.finished_at = datetime.utcnow().isoformat()
                    session.add(db_job)
                    session.commit()

            # Dispatch Failure Webhook
            webhook_url = job.webhook_url or user.webhook_url
            if webhook_url:
                send_webhook(webhook_url, {
                    "job_id": job_id,
                    "status": "failed",
                    "filename": job.filename,
                    "transcript": None,
                    "llm_summary": None,
                    "error": error_msg
                })
        
        # Ensure temporary file is always cleaned up after processing (success or permanent failure)
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as del_err:
            logger.warning(f"Failed to delete temp file {file_path}: {del_err}")

def idle_shutdown_monitor_loop(stop_event: threading.Event):
    """Tracks inactivity and unassigns idle runtimes to free GPU quotas."""
    logger.info("Starting idle Colab instance shutdown monitor...")
    while not stop_event.is_set():
        stop_event.wait(30.0)
        with active_runtimes_lock:
            for user_id in list(active_runtimes.keys()):
                info = active_runtimes[user_id]
                
                with Session(database.engine) as session:
                    # Count running/pending jobs
                    active_count = len(session.exec(select(Job).where(
                        Job.user_id == user_id,
                        Job.status.in_(["pending", "starting_colab", "uploading", "processing"])
                    )).all())

                if active_count > 0:
                    info["last_active"] = time.time()
                    continue

                idle_sec = time.time() - info["last_active"]
                if idle_sec > config.settings.idle_timeout_sec:
                    logger.info(f"User {user_id} idle for {idle_sec:.1f}s. Tearing down Colab runtime...")
                    try:
                        info["runtime"].close()
                        info["client"].unassign_runtime(info["runtime"].assignment.endpoint)
                    except Exception as e:
                        logger.error(f"Failed to unassign runtime for user {user_id}: {e}")
                    finally:
                        del active_runtimes[user_id]

def cleanup_active_runtimes():
    """Unassigns all active Google Colab runtimes during server shutdown."""
    logger.info("Cleaning up active Colab runtimes...")
    with active_runtimes_lock:
        for user_id, info in list(active_runtimes.items()):
            try:
                info["runtime"].close()
                info["client"].unassign_runtime(info["runtime"].assignment.endpoint)
            except Exception as e:
                logger.error(f"Error clean closing runtime for user {user_id}: {e}")
        active_runtimes.clear()
