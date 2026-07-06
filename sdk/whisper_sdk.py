from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    import sys
    sys.exit("Install requests:  pip install requests")

_SPINNER = ["|", "/", "–", "\\"]
_BAR_LEN = 34


def _progress_bar(pct: int, spin: int) -> str:
    filled = int(_BAR_LEN * pct / 100)
    bar = "█" * filled + "░" * (_BAR_LEN - filled)
    return f"[{bar}] {pct:3d}%  {_SPINNER[spin % len(_SPINNER)]}"


def _sizeof_fmt(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num //= 1024
    return f"{num:.1f} TB"


class WhisperClient:

    def __init__(self, server_url: str, timeout: int = 120, upload_timeout: int = 600, poll_interval: float = 3.0,
            verbose: bool = True, ):
        self.base = server_url.rstrip("/")
        self.timeout = timeout
        self.upload_timeout = upload_timeout
        self.poll_interval = poll_interval
        self.verbose = verbose

        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1.0, status_forcelist=[502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        self._s = session

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        r = self._s.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        self._raise(r)
        return r.json()

    def _post(self, path: str, **kw) -> Dict:
        timeout = kw.pop("_timeout", self.timeout)
        r = self._s.post(f"{self.base}{path}", timeout=timeout, **kw)
        self._raise(r)
        return r.json()

    def _delete(self, path: str) -> Dict:
        r = self._s.delete(f"{self.base}{path}", timeout=self.timeout)
        self._raise(r)
        return r.json()

    @staticmethod
    def _raise(r: requests.Response) -> None:
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:400]
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r) from exc

    def _vprint(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def submit(self, audio_path: Optional[str] = None, gdrive_id: Optional[str] = None, *,
            model: str = "large-v3-turbo", language: Optional[str] = None, task: str = "transcribe", beam_size: int = 5,
            best_of: int = 5, patience: float = 1.0, temperature: Union[float, list] = 0.0,
            compression_ratio_threshold: float = 2.4, log_prob_threshold: float = -1.0,
            no_speech_threshold: float = 0.6, condition_on_previous_text: bool = True, word_timestamps: bool = True,
            vad_filter: bool = True, vad_min_silence_duration_ms: int = 500, initial_prompt: Optional[str] = None,
            hotwords: Optional[str] = None, repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
            hallucination_silence_threshold: Optional[float] = None, max_new_tokens: Optional[int] = None,
            without_timestamps: bool = False, prepend_punctuations: str = "\"'\u2018\u2019\u00bf([{-",
            append_punctuations: str = "\"'\u2019.`\uff0c,\uff01!\uff1f?::\")}]\u3001", ) -> str:

        if audio_path is None and gdrive_id is None:
            raise ValueError("Provide either audio_path or gdrive_id.")
        if audio_path is not None and gdrive_id is not None:
            raise ValueError("Provide only one of audio_path or gdrive_id.")

        if isinstance(temperature, (list, tuple)):
            temp_str = ",".join(str(t) for t in temperature)
        else:
            temp_str = str(float(temperature))

        data: Dict = {"model": model, "task": task, "beam_size": str(beam_size), "best_of": str(best_of),
            "patience": str(patience), "temperature": temp_str,
            "compression_ratio_threshold": str(compression_ratio_threshold),
            "log_prob_threshold": str(log_prob_threshold), "no_speech_threshold": str(no_speech_threshold),
            "condition_on_previous_text": "true" if condition_on_previous_text else "false",
            "word_timestamps": "true" if word_timestamps else "false", "vad_filter": "true" if vad_filter else "false",
            "vad_min_silence_duration_ms": str(vad_min_silence_duration_ms),
            "repetition_penalty": str(repetition_penalty), "no_repeat_ngram_size": str(no_repeat_ngram_size),
            "without_timestamps": "true" if without_timestamps else "false",
            "prepend_punctuations": prepend_punctuations, "append_punctuations": append_punctuations, }
        if language:
            data["language"] = language
        if initial_prompt:
            data["initial_prompt"] = initial_prompt
        if hotwords:
            data["hotwords"] = hotwords
        if hallucination_silence_threshold is not None:
            data["hallucination_silence_threshold"] = str(hallucination_silence_threshold)
        if max_new_tokens is not None:
            data["max_new_tokens"] = str(max_new_tokens)
        if gdrive_id:
            data["gdrive_id"] = gdrive_id

        files = None
        if audio_path is not None:
            p = Path(audio_path)
            if not p.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            size = p.stat().st_size
            self._vprint(f"[Whisper] Uploading {p.name}  ({_sizeof_fmt(size)}) …")
            files = {"audio": (p.name, open(str(p), "rb"), "application/octet-stream")}
        else:
            self._vprint(f"[Whisper] Requesting server to fetch GDrive file: {gdrive_id} …")

        try:
            resp = self._post("/transcribe", data=data, files=files, _timeout=self.upload_timeout, )
        finally:
            if files:
                for _, fh, _ in files.values():
                    if hasattr(fh, "close"):
                        fh.close()

        job_id = resp["job_id"]
        self._vprint(f"[Whisper] Job submitted: {job_id}  "
                     f"(model={resp.get('model', '?')}, task={resp.get('task', '?')})")
        return job_id

    def poll(self, job_id: str, since_log: int = 0) -> Dict:

        return self._get(f"/jobs/{job_id}", params={"since_log": since_log})

    def wait(self, job_id: str) -> Dict:

        log_offset = 0
        spin = 0
        last_pct = -1

        while True:
            status = self.poll(job_id, since_log=log_offset)
            new_logs = status.get("logs", [])
            log_offset = status.get("log_offset", log_offset)

            if self.verbose:
                for line in new_logs:
                    print(f"\r\033[K  {line}")

            pct = status.get("progress", 0)
            if self.verbose and pct != last_pct:
                bar = _progress_bar(pct, spin)
                print(f"\r  {bar}", end="", flush=True)
                spin += 1
                last_pct = pct

            st = status["status"]

            if st == "done":
                if self.verbose:
                    print()
                return status["result"]

            if st == "failed":
                if self.verbose:
                    print()
                raise RuntimeError(f"Job {job_id} failed:\n"
                                   f"{status.get('error', '(no details)')}")

            if st == "cancelled":
                if self.verbose:
                    print()
                raise RuntimeError(f"Job {job_id} was cancelled.")

            time.sleep(self.poll_interval)
