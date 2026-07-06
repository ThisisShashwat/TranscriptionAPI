from __future__ import annotations

PORT = 8000
DEFAULT_MODEL = "small"
COMPUTE_TYPE = "float16"
CPU_THREADS = 4
NUM_WORKERS = 2
MAX_CONCURRENT_JOBS = 2
PRELOAD_MODEL = True


import concurrent.futures
import logging
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment, TranscriptionInfo, Word
except ImportError:
    sys.exit("\nMissing dependency: faster-whisper\n\n"
             "Run in Colab:\n"
             "  !pip install faster-whisper fastapi 'uvicorn[standard]' "
             "gdown python-multipart\n")

try:
    import uvicorn
    from fastapi import (FastAPI, File, Form, HTTPException, Query, UploadFile, )
    from fastapi.middleware.cors import CORSMiddleware
    import re
except ImportError as exc:
    sys.exit(f"\nMissing dependency: {exc}\n")

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import gdown

    _GDOWN_AVAILABLE = True
except ImportError:
    _GDOWN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)-8s]  %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)], force=True, )
log = logging.getLogger("whisper.server")

VALID_MODELS = ["large-v3-turbo", "large-v3", "distil-large-v3", "medium", "medium.en", "small", "small.en", "base",
    "base.en", "tiny", "tiny.en", ]

VALID_TASKS = ["transcribe", "translate"]


class Config:
    DEFAULT_MODEL: str = "small"
    COMPUTE_TYPE: str = "float16"
    CPU_THREADS: int = 4
    NUM_WORKERS: int = 2
    MAX_CONCURRENT: int = 2
    TEMP_DIR: Path = Path("_tmp_whisper")


cfg = Config()

_model_cache: Dict[str, WhisperModel] = {}
_model_lock = threading.Lock()


def _best_device() -> str:
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _effective_compute_type(device: str) -> str:
    if device == "cpu" and cfg.COMPUTE_TYPE == "float16":
        return "int8"
    return cfg.COMPUTE_TYPE


def get_model(model_name: str) -> WhisperModel:
    with _model_lock:
        if model_name not in _model_cache:
            device = _best_device()
            ctype = _effective_compute_type(device)
            log.info(f"Loading model '{model_name}' on {device} ({ctype}) …")
            _model_cache[model_name] = WhisperModel(model_name, device=device, compute_type=ctype,
                cpu_threads=cfg.CPU_THREADS, num_workers=cfg.NUM_WORKERS, )
            log.info(f"Model '{model_name}' loaded ✓")
    return _model_cache[model_name]


def list_loaded_models() -> List[str]:
    with _model_lock:
        return list(_model_cache.keys())


def unload_model(model_name: str) -> bool:
    with _model_lock:
        if model_name in _model_cache:
            del _model_cache[model_name]
            log.info(f"Model '{model_name}' unloaded.")
            return True
    return False


_executor: concurrent.futures.ThreadPoolExecutor


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utcnow_short() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


class Job:

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "queued"
        self.progress = 0
        self.logs: List[str] = []
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.created_at = _utcnow()
        self.finished_at: Optional[str] = None
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

    def _append_log(self, line: str) -> None:
        with self._lock:
            self.logs.append(line)

    def log_info(self, msg: str) -> None:
        line = f"[{_utcnow_short()}] [INFO ] {msg}"
        log.info(msg)
        self._append_log(line)

    def log_warn(self, msg: str) -> None:
        line = f"[{_utcnow_short()}] [WARN ] {msg}"
        log.warning(msg)
        self._append_log(line)

    def log_error(self, msg: str) -> None:
        line = f"[{_utcnow_short()}] [ERROR] {msg}"
        log.error(msg)
        self._append_log(line)

    def set_progress(self, pct: int, msg: str = "") -> None:
        self.progress = min(100, max(0, pct))
        if msg:
            self.log_info(msg)

    def finish(self, result: Dict) -> None:
        with self._lock:
            self.result = result
            self.status = "done"
            self.progress = 100
            self.finished_at = _utcnow()

    def fail(self, err: str) -> None:
        with self._lock:
            self.error = err
            self.status = "failed"
            self.finished_at = _utcnow()
        log.error(f"Job {self.job_id} FAILED:\n{err}")

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def as_dict(self, since_log: int = 0) -> Dict:
        with self._lock:
            logs_slice = list(self.logs[since_log:])
            log_offset = since_log + len(logs_slice)
            return {"job_id": self.job_id, "status": self.status, "progress": self.progress, "logs": logs_slice,
                "log_offset": log_offset, "total_logs": len(self.logs), "result": self.result, "error": self.error,
                "created_at": self.created_at, "finished_at": self.finished_at, }


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()


def new_job() -> Job:
    j = Job(str(uuid.uuid4()))
    with _jobs_lock:
        _jobs[j.job_id] = j
    log.info(f"New job: {j.job_id}")
    return j


def get_job_or_404(job_id: str) -> Job:
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None:
        raise HTTPException(404, detail=f"Job '{job_id}' not found.")
    return j


def _ffmpeg(src: str, dst: str, extra_args: Optional[List[str]] = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src]
    cmd += extra_args or []
    cmd.append(dst)
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {r.returncode}):\n"
                           f"  cmd: {' '.join(cmd)}\n"
                           f"  stderr: {r.stderr.decode(errors='replace')}")


def preprocess_to_wav(src: str, job: Optional[Job] = None) -> str:
    stem = Path(src).stem
    dst = str(cfg.TEMP_DIR / f"{stem}_{uuid.uuid4().hex[:8]}.wav")

    msg = f"Pre-processing {Path(src).name} → 16 kHz mono WAV …"
    if job:
        job.log_info(msg)
    else:
        log.info(msg)

    _ffmpeg(src, dst, extra_args=["-ac", "1", "-ar", "16000", "-sample_fmt", "s16"])

    size_mb = Path(dst).stat().st_size / 1_048_576
    msg2 = f"Pre-processing done → {Path(dst).name}  ({size_mb:.1f} MB)"
    if job:
        job.log_info(msg2)
    else:
        log.info(msg2)

    return dst


def audio_duration_sec(wav_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            wav_path, ], capture_output=True, )
    try:
        return float(r.stdout.decode().strip())
    except (ValueError, AttributeError):

        return 0.0


def gdrive_download(file_id: str, job: Job) -> str:
    if not _GDOWN_AVAILABLE:
        raise RuntimeError("gdown is not installed.  pip install gdown")
    out = str(cfg.TEMP_DIR / f"gdrive_{file_id[:12]}_{uuid.uuid4().hex[:6]}.audio")
    job.log_info(f"Downloading from Google Drive (id={file_id}) …")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out, quiet=False)
    if not Path(out).exists():
        raise RuntimeError(f"GDrive download failed for file_id={file_id!r}. "
                           "Ensure the file is shared as 'Anyone with the link can view'.")
    size_mb = Path(out).stat().st_size / 1_048_576
    job.log_info(f"GDrive download complete: {Path(out).name}  ({size_mb:.1f} MB)")
    return out


import math


def _safe_float(value, ndigits: int = 6):
    if value is None:
        return None
    f = float(value)
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _word_to_dict(w: Word) -> Dict:
    return {"word": w.word, "start": _safe_float(w.start, 3), "end": _safe_float(w.end, 3),
        "probability": _safe_float(w.probability, 6), }


def _segment_to_dict(seg: Segment, index: int) -> Dict:
    words = None
    if seg.words:
        words = [_word_to_dict(w) for w in seg.words]

    return {

        "index": index, "id": seg.id, "seek": seg.seek, "start": _safe_float(seg.start, 3),
        "end": _safe_float(seg.end, 3), "duration": _safe_float(seg.end - seg.start, 3),

        "text": seg.text, "tokens": list(seg.tokens),

        "temperature": _safe_float(seg.temperature, 4), "avg_logprob": _safe_float(seg.avg_logprob, 6),
        "compression_ratio": _safe_float(seg.compression_ratio, 6),
        "no_speech_prob": _safe_float(seg.no_speech_prob, 6),

        "words": words, }


def _transcription_options_to_dict(opts) -> Dict:
    if opts is None:
        return {}

    def _safe_val(attr, default=None):
        v = getattr(opts, attr, default)
        if v is None:
            return default

        if isinstance(v, (tuple, list)):
            return [_safe_float(x) if isinstance(x, float) else x for x in v]

        if isinstance(v, float):
            return _safe_float(v)
        return v

    return {"beam_size": _safe_val("beam_size"), "best_of": _safe_val("best_of"), "patience": _safe_val("patience"),
        "length_penalty": _safe_val("length_penalty"), "repetition_penalty": _safe_val("repetition_penalty"),
        "no_repeat_ngram_size": _safe_val("no_repeat_ngram_size"),
        "log_prob_threshold": _safe_val("log_prob_threshold"), "no_speech_threshold": _safe_val("no_speech_threshold"),
        "compression_ratio_threshold": _safe_val("compression_ratio_threshold"),
        "temperatures": _safe_val("temperatures"), "initial_prompt": _safe_val("initial_prompt"),
        "prefix": _safe_val("prefix"), "suppress_blank": _safe_val("suppress_blank"),
        "suppress_tokens": _safe_val("suppress_tokens"), "without_timestamps": _safe_val("without_timestamps"),
        "max_initial_timestamp": _safe_val("max_initial_timestamp"), "word_timestamps": _safe_val("word_timestamps"),
        "prepend_punctuations": _safe_val("prepend_punctuations"),
        "append_punctuations": _safe_val("append_punctuations"), "multilingual": _safe_val("multilingual"),
        "output_language": _safe_val("output_language"), "max_new_tokens": _safe_val("max_new_tokens"),
        "clip_timestamps": _safe_val("clip_timestamps"),
        "hallucination_silence_threshold": _safe_val("hallucination_silence_threshold"),
        "hotwords": _safe_val("hotwords"), "language_detection_threshold": _safe_val("language_detection_threshold"),
        "language_detection_segments": _safe_val("language_detection_segments"), }


def _vad_options_to_dict(opts) -> Dict:
    if opts is None:
        return {}

    def _safe_val(attr, default=None):
        v = getattr(opts, attr, default)
        return _safe_float(v) if isinstance(v, float) else v

    return {"vad_onset": _safe_val("vad_onset"), "vad_offset": _safe_val("vad_offset"),
        "min_speech_duration_ms": _safe_val("min_speech_duration_ms"),
        "max_speech_duration_s": _safe_val("max_speech_duration_s"),
        "min_silence_duration_ms": _safe_val("min_silence_duration_ms"), "speech_pad_ms": _safe_val("speech_pad_ms"), }


def _info_to_dict(info: TranscriptionInfo) -> Dict:
    all_lang_probs = None
    if info.all_language_probs:
        all_lang_probs = [{"language": lang, "probability": _safe_float(prob, 6)} for lang, prob in
            info.all_language_probs]

    return {"language": info.language, "language_probability": _safe_float(info.language_probability, 6),
        "duration": _safe_float(info.duration, 3), "duration_after_vad": _safe_float(info.duration_after_vad, 3),
        "all_language_probs": all_lang_probs,
        "transcription_options": _transcription_options_to_dict(info.transcription_options),
        "vad_options": _vad_options_to_dict(info.vad_options), }


def _compute_segment_stats(segments: List[Dict]) -> Dict:
    if not segments:
        return {"avg_logprob": None, "avg_no_speech_prob": None, "avg_compression_ratio": None, "avg_temperature": None,
            "min_logprob": None, "max_no_speech_prob": None, "segments_with_words": 0, "total_words": 0, }

    logprobs = [s["avg_logprob"] for s in segments if s["avg_logprob"] is not None]
    no_speech = [s["no_speech_prob"] for s in segments if s["no_speech_prob"] is not None]
    comp_ratio = [s["compression_ratio"] for s in segments if s["compression_ratio"] is not None]
    temps = [s["temperature"] for s in segments if s["temperature"] is not None]
    n_with_words = sum(1 for s in segments if s.get("words"))
    total_words = sum(len(s["words"]) for s in segments if s.get("words"))

    return {"avg_logprob": round(sum(logprobs) / len(logprobs), 6) if logprobs else None,
        "avg_no_speech_prob": round(sum(no_speech) / len(no_speech), 6) if no_speech else None,
        "avg_compression_ratio": round(sum(comp_ratio) / len(comp_ratio), 6) if comp_ratio else None,
        "avg_temperature": round(sum(temps) / len(temps), 6) if temps else None,
        "min_logprob": round(min(logprobs), 6) if logprobs else None,
        "max_no_speech_prob": round(max(no_speech), 6) if no_speech else None, "segments_with_words": n_with_words,
        "total_words": total_words, }


def process_job(job: Job, wav_path: str, original_filename: str, file_size_bytes: int,

        model_name: str, language: Optional[str], task: str, beam_size: int, best_of: int, patience: float,
        temperature: Union[float, Tuple[float, ...]], compression_ratio_threshold: float, log_prob_threshold: float,
        no_speech_threshold: float, condition_on_previous_text: bool, word_timestamps: bool, vad_filter: bool,
        vad_min_silence_duration_ms: int, initial_prompt: Optional[str], hotwords: Optional[str],
        repetition_penalty: float, no_repeat_ngram_size: int, hallucination_silence_threshold: Optional[float],
        max_new_tokens: Optional[int], without_timestamps: bool, prepend_punctuations: str,
        append_punctuations: str, ) -> None:
    t_total = time.time()

    try:
        job.status = "running"
        job.set_progress(3, f"Loading model '{model_name}' …")

        if job.is_cancelled():
            job.fail("Cancelled before model load.")
            return

        model = get_model(model_name)
        device = _best_device()

        job.set_progress(12, f"Model ready on {device}.  Starting transcription …")

        duration = audio_duration_sec(wav_path)
        duration_min = duration / 60 if duration > 0 else 0
        job.log_info(f"Audio: {original_filename}  ({duration:.1f}s / {duration_min:.1f} min)")

        transcribe_kwargs: Dict[str, Any] = {"beam_size": beam_size, "best_of": best_of, "patience": patience,
            "temperature": temperature, "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold, "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text, "word_timestamps": word_timestamps,
            "vad_filter": vad_filter, "vad_parameters": {"min_silence_duration_ms": vad_min_silence_duration_ms},
            "task": task, "repetition_penalty": repetition_penalty, "no_repeat_ngram_size": no_repeat_ngram_size,
            "without_timestamps": without_timestamps, "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations, }

        if language:
            transcribe_kwargs["language"] = language
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if hotwords:
            transcribe_kwargs["hotwords"] = hotwords
        if hallucination_silence_threshold is not None:
            transcribe_kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
        if max_new_tokens is not None:
            transcribe_kwargs["max_new_tokens"] = max_new_tokens

        job.set_progress(15, "Transcribing …  (progress updates every ~10 segments)")

        seg_generator, info = model.transcribe(wav_path, **transcribe_kwargs)

        segments_raw: List[Segment] = []
        for seg in seg_generator:
            if job.is_cancelled():
                job.fail("Job cancelled during transcription.")
                return

            segments_raw.append(seg)

            if duration > 0 and len(segments_raw) % 10 == 0:
                audio_pos = seg.end
                pct = 15 + int(70 * min(audio_pos / duration, 1.0))
                job.set_progress(pct, f"  … {seg.end:.1f}s / {duration:.1f}s  ({len(segments_raw)} segs)")

        job.set_progress(88, f"Transcription done.  {len(segments_raw)} segment(s).  Building result …")

        segments_dict = [_segment_to_dict(s, i) for i, s in enumerate(segments_raw)]
        info_dict = _info_to_dict(info)
        seg_stats = _compute_segment_stats(segments_dict)
        full_text = " ".join(s["text"].strip() for s in segments_dict)

        job.set_progress(95, "Assembling final result …")

        result: Dict[str, Any] = {

            "job_id": job.job_id, "status": "success", "file_name": original_filename,
            "file_size_bytes": file_size_bytes,

            "duration_seconds": round(duration, 3), "duration_minutes": round(duration_min, 3),

            "model": model_name, "compute_type": _effective_compute_type(device), "device": device, "task": task,
            "language_requested": language,

            "detected_language": info_dict["language"],
            "detected_language_probability": info_dict["language_probability"],
            "duration_after_vad": info_dict["duration_after_vad"],
            "all_language_probs": info_dict["all_language_probs"],

            "transcription_options": info_dict["transcription_options"], "vad_options": info_dict["vad_options"],

            "full_text": full_text, "total_segments": len(segments_dict), "total_words": seg_stats["total_words"],

            "segment_stats": {"avg_logprob": seg_stats["avg_logprob"],
                "avg_no_speech_prob": seg_stats["avg_no_speech_prob"],
                "avg_compression_ratio": seg_stats["avg_compression_ratio"],
                "avg_temperature": seg_stats["avg_temperature"], "min_logprob": seg_stats["min_logprob"],
                "max_no_speech_prob": seg_stats["max_no_speech_prob"],
                "segments_with_words": seg_stats["segments_with_words"], },

            "segments": segments_dict,

            "processing_time_seconds": round(time.time() - t_total, 2),
            "realtime_factor": round((time.time() - t_total) / duration, 3) if duration > 0 else None,
            "finished_at": _utcnow(), }

        job.finish(result)
        job.log_info(f"✓ Done  |  {len(segments_dict)} segments  |  "
                     f"{seg_stats['total_words']} words  |  "
                     f"lang={info_dict['language']} ({info_dict['language_probability']:.2%})  |  "
                     f"{result['processing_time_seconds']:.1f}s")

    except Exception:
        job.fail(traceback.format_exc())
    finally:
        try:
            if wav_path and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass


app = FastAPI(title="Faster-Whisper Transcription Server", version="1.0.0",
    description=("GPU-accelerated speech transcription via faster-whisper.  "
                 "Submit jobs, poll for progress, retrieve full segment-level output."), )

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], )


@app.get("/health", tags=["meta"])
def health():
    with _jobs_lock:
        n_running = sum(1 for j in _jobs.values() if j.status == "running")
        n_queued = sum(1 for j in _jobs.values() if j.status == "queued")
        n_total = len(_jobs)

    gpu_info: Optional[Dict] = None
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {"name": props.name, "total_memory_gb": round(props.total_memory / 1e9, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3), }

    return {"status": "ok", "server_version": "1.0.0", "device": _best_device(),
        "cuda_available": _TORCH_AVAILABLE and torch.cuda.is_available(), "gpu_info": gpu_info,
        "loaded_models": list_loaded_models(), "default_model": cfg.DEFAULT_MODEL, "compute_type": cfg.COMPUTE_TYPE,
        "max_concurrent_jobs": cfg.MAX_CONCURRENT, "active_jobs": n_running, "queued_jobs": n_queued,
        "total_jobs": n_total, }


@app.get("/models", tags=["meta"])
def models():
    return {"available_models": VALID_MODELS, "loaded_models": list_loaded_models(),
        "default_model": cfg.DEFAULT_MODEL, }


@app.delete("/models/{model_name}", tags=["meta"])
def unload_model_endpoint(model_name: str):
    ok = unload_model(model_name)
    if not ok:
        raise HTTPException(404, detail=f"Model '{model_name}' is not loaded.")
    return {"unloaded": model_name, "status": "ok"}


def _parse_temperature(val: str) -> Union[float, Tuple[float, ...]]:
    parts = [p.strip() for p in val.split(",") if p.strip()]
    floats = [float(p) for p in parts]
    return floats[0] if len(floats) == 1 else tuple(floats)


@app.post("/transcribe", tags=["transcription"])
async def start_transcribe(

        audio: Optional[UploadFile] = File(None, description="Audio file upload (any format)"),
        gdrive_id: Optional[str] = Form(None, description="Google Drive file ID (server downloads it)"),

        model: str = Form("large-v3-turbo",
                          description="Model name: large-v3-turbo | large-v3 | distil-large-v3 | medium | base"),
        language: Optional[str] = Form(None, description="ISO-639-1 language code (e.g. 'en', 'fr').  "
                                                         "Omit for auto-detect."),
        task: str = Form("transcribe", description="'transcribe' (keep source language) or 'translate' (→ English)"),

        beam_size: int = Form(5, description="Beam search width (higher = more accurate, slower)"),
        best_of: int = Form(5, description="Candidates for non-zero temperature sampling"),
        patience: float = Form(1.0, description="Beam search patience factor"),
        temperature: str = Form("0", description="Sampling temperature.  Single float or comma-sep list "
                                                 "for fallback (e.g. '0,0.2,0.4,0.6,0.8,1.0')"),
        compression_ratio_threshold: float = Form(2.4, description="Max compression ratio before fallback"),
        log_prob_threshold: float = Form(-1.0, description="Min avg log-prob before fallback"),
        no_speech_threshold: float = Form(0.6, description="no_speech_prob above this → segment is silence"),
        condition_on_previous_text: str = Form("true", description="Use previous output as prompt for next window"),
        repetition_penalty: float = Form(1.0, description="Penalise repeated tokens (1.0 = off)"),
        no_repeat_ngram_size: int = Form(0, description="Block repeated n-grams (0 = off)"),
        hallucination_silence_threshold: Optional[float] = Form(None,
                                                                description="Skip silent segments > this many seconds (None = off)"),
        max_new_tokens: Optional[int] = Form(None, description="Hard cap on tokens generated per window"),

        word_timestamps: str = Form("true", description="Return per-word start/end/probability"),
        without_timestamps: str = Form("false", description="Disable all timestamp prediction (faster for pure text)"),
        prepend_punctuations: str = Form("\"\u2018\u2019\u00bf([{-",
                                         description="Punctuation glued to the START of the following word"),
        append_punctuations: str = Form("\"\u2019.`\uff0c,\uff01!\uff1f?::\")]}\u3001",
                                        description="Punctuation glued to the END of the preceding word"),

        vad_filter: str = Form("true", description="Strip non-speech with Silero VAD before decoding"),
        vad_min_silence_duration_ms: int = Form(500, description="Minimum silence gap (ms) to split on with VAD"),

        initial_prompt: Optional[str] = Form(None, description="Text prepended as context to the first window"),
        hotwords: Optional[str] = Form(None, description="Comma-separated words to boost recognition of"), ):
    if audio is None and not gdrive_id:
        raise HTTPException(400, detail="Provide either an `audio` file upload or a `gdrive_id`.")
    if audio is not None and gdrive_id:
        raise HTTPException(400, detail="Provide only one of `audio` or `gdrive_id`.")

    model = model.strip() or cfg.DEFAULT_MODEL
    if model not in VALID_MODELS:
        raise HTTPException(400, detail=f"Unknown model '{model}'.  Valid: {VALID_MODELS}")

    task = task.strip().lower()
    if task not in VALID_TASKS:
        raise HTTPException(400, detail=f"task must be one of {VALID_TASKS}.")

    def _bool(v: str) -> bool:
        return str(v).lower() in ("1", "true", "yes")

    try:
        temp_parsed = _parse_temperature(temperature)
    except ValueError:
        raise HTTPException(400, detail=f"Invalid temperature value: '{temperature}'")

    job = new_job()
    raw_path: Optional[str] = None
    original_filename: str = f"gdrive_{gdrive_id}" if gdrive_id else "unknown"
    file_size_bytes: int = 0

    if audio is not None:
        suffix = Path(audio.filename).suffix or ".audio"
        raw_path = str(cfg.TEMP_DIR / f"upload_{job.job_id}{suffix}")
        data = await audio.read()
        file_size_bytes = len(data)
        with open(raw_path, "wb") as f:
            f.write(data)
        original_filename = audio.filename
        job.log_info(f"Upload received: {audio.filename}  ({file_size_bytes / 1_048_576:.1f} MB)")

    params = dict(model_name=model, language=language.strip() if language else None, task=task, beam_size=beam_size,
        best_of=best_of, patience=patience, temperature=temp_parsed,
        compression_ratio_threshold=compression_ratio_threshold, log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold, condition_on_previous_text=_bool(condition_on_previous_text),
        word_timestamps=_bool(word_timestamps), vad_filter=_bool(vad_filter),
        vad_min_silence_duration_ms=vad_min_silence_duration_ms, initial_prompt=initial_prompt or None,
        hotwords=hotwords or None, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        hallucination_silence_threshold=hallucination_silence_threshold, max_new_tokens=max_new_tokens,
        without_timestamps=_bool(without_timestamps), prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations, )

    def _run_in_background():
        nonlocal raw_path
        try:
            if gdrive_id:
                raw_path = gdrive_download(gdrive_id, job)

            wav_path = preprocess_to_wav(raw_path, job)

            if raw_path and Path(raw_path).exists() and raw_path != wav_path:
                Path(raw_path).unlink(missing_ok=True)

            process_job(job=job, wav_path=wav_path, original_filename=original_filename,
                file_size_bytes=file_size_bytes, **params, )
        except Exception:
            job.fail(traceback.format_exc())

    _executor.submit(_run_in_background)

    return {"job_id": job.job_id, "status": "queued", "original_file": original_filename, "model": model, "task": task,
        "poll_url": f"/jobs/{job.job_id}", "message": "Job submitted.  Poll /jobs/{job_id} for progress and results.", }


@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str, since_log: int = Query(0, description="Return only log lines from this offset onward"), ):
    j = get_job_or_404(job_id)
    return j.as_dict(since_log=since_log)


@app.post("/jobs/{job_id}/cancel", tags=["jobs"])
def cancel_job(job_id: str):
    j = get_job_or_404(job_id)
    j.cancel()
    return {"job_id": job_id, "cancelled": True, "note": "Cancellation is best-effort."}


@app.get("/jobs", tags=["jobs"])
def list_jobs():
    with _jobs_lock:
        return [{"job_id": j.job_id, "status": j.status, "progress": j.progress, "created_at": j.created_at,
            "finished_at": j.finished_at, } for j in sorted(_jobs.values(), key=lambda x: x.created_at, reverse=True)]


public_url = ""

@app.on_event("startup")
def print_banner_on_startup():
    global public_url
    border = "=" * 66
    log.info(border)
    log.info("  Faster-Whisper Transcription Server  v1.0.0")
    log.info(f"  Public URL   :  {public_url}")
    log.info(f"  Local URL    :  http://localhost:{PORT}")
    log.info(f"  API Docs     :  {public_url}/docs")
    log.info(f"  Device       :  {_best_device()}")
    log.info(f"  Default model:  {DEFAULT_MODEL}  ({COMPUTE_TYPE})")
    log.info(f"  Max parallel :  {MAX_CONCURRENT_JOBS} job(s)")
    log.info(border)
    print(f"\n>>> SERVER URL (copy into whisper_sdk.py): {public_url}\n", flush=True)

def _startup():
    global _executor, public_url

    cfg.DEFAULT_MODEL = DEFAULT_MODEL
    cfg.COMPUTE_TYPE = COMPUTE_TYPE
    cfg.CPU_THREADS = CPU_THREADS
    cfg.NUM_WORKERS = NUM_WORKERS
    cfg.MAX_CONCURRENT = MAX_CONCURRENT_JOBS
    cfg.TEMP_DIR.mkdir(exist_ok=True)

    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS,
        thread_name_prefix="whisper-worker", )

    cf_proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    for line in cf_proc.stderr:
        line = line.decode()
        match = re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
        if match:
            public_url = match.group(0)
            break

    if PRELOAD_MODEL:
        log.info(f"Preloading '{DEFAULT_MODEL}' (PRELOAD_MODEL=True) …")
        get_model(DEFAULT_MODEL)


import nest_asyncio

nest_asyncio.apply()
if __name__ == "__main__":
    _startup()
    import asyncio

    loop = asyncio.get_event_loop()
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
