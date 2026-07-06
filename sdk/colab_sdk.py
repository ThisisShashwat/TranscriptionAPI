from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install requests") from exc

try:
    import websocket as _websocket_module
except ImportError as exc:
    _websocket_module = None
    _WEBSOCKET_IMPORT_ERROR: Exception | None = exc
else:
    _WEBSOCKET_IMPORT_ERROR = None

COLAB_DOMAIN = "https://colab.research.google.com"
COLAB_GAPI_DOMAIN = "https://colab.pa.googleapis.com"
EXTENSION_VERSION = "0.8.1"
JSON_PREFIX = ")]}'\n"

DEFAULT_SCOPES = ["profile", "email", "https://www.googleapis.com/auth/colaboratory"]

DEFAULT_ACCOUNTS_PATH = "colab_accounts.json"

_OUTCOME_QUOTA = (1, 2)
_OUTCOME_BLOCKED = (5,)


class ColabError(RuntimeError):
    """General Colab SDK error."""


class ColabQuotaError(ColabError):
    """Raised when Colab reports insufficient quota (outcome 1 or 2)."""


class ColabBlockedError(ColabError):
    """Raised when Colab reports the account is blocked (outcome 5)."""


class ColabRuntimeDeadError(ColabError):
    """Raised when a runtime is no longer reachable."""


class ColabTokenError(ColabError):
    """Raised when an OAuth token is missing or cannot be refreshed."""


@dataclass
class AccountProfile:
    name: str
    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""
    access_token: str = ""
    email: str = ""
    scopes: list[str] = field(default_factory=lambda: list(DEFAULT_SCOPES))


@dataclass
class RuntimeAssignment:
    assignment_id: str
    endpoint: str
    base_url: str
    runtime_token: str
    variant: str = "DEFAULT"
    accelerator: str = ""
    highmem: bool = False
    assigned_at: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        parts = [self.variant or "DEFAULT"]
        if self.accelerator:
            parts.append(self.accelerator)
        if self.highmem:
            parts.append("HIGHMEM")
        return " ".join(parts)



@dataclass
class CellExecution:
    index: int
    code: str
    execution_count: int | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    errored: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def text(self) -> str:
        return _cell_text(self.outputs)

    def tail(self, n: int = 20) -> str:
        lines = [ln for ln in self.text.splitlines() if ln]
        return "\n".join(lines[-n:])


class TokenProvider:
    _TOKEN_EXPIRY_BUFFER = 300

    def __init__(self, *, access_token: str | None = None, refresh_token: str | None = None,
            client_id: str | None = None, client_secret: str | None = None):
        self._access_token = access_token or os.environ.get("COLAB_ACCESS_TOKEN") or ""
        self._refresh_token = refresh_token or os.environ.get("COLAB_REFRESH_TOKEN") or os.environ.get(
            "GOOGLE_REFRESH_TOKEN") or ""
        self._client_id = client_id or os.environ.get("GOOGLE_CLIENT_ID") or ""
        self._client_secret = client_secret or os.environ.get("GOOGLE_CLIENT_SECRET") or ""
        self._expires_at: float = 0.0

    def get_access_token(self) -> str:
        if self._access_token and self._expires_at:
            if time.time() < self._expires_at - self._TOKEN_EXPIRY_BUFFER:
                return self._access_token
        elif self._access_token:
            return self._access_token
        return self._refresh()

    def _refresh(self) -> str:
        if not self._refresh_token or not self._client_id:
            raise ColabTokenError("No Colab token configured. Set COLAB_ACCESS_TOKEN, or provide "
                                  "GOOGLE_CLIENT_ID + COLAB_REFRESH_TOKEN / GOOGLE_REFRESH_TOKEN.")
        data: dict[str, str] = {"client_id": self._client_id, "refresh_token": self._refresh_token,
            "grant_type": "refresh_token"}
        if self._client_secret:
            data["client_secret"] = self._client_secret
        resp = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=30)
        payload = _checked_json(resp)
        token = payload.get("access_token")
        if not token:
            raise ColabTokenError(f"Refresh response missing access_token: {payload}")
        self._access_token = token
        expires_in = int(payload.get("expires_in", 0) or 0)
        self._expires_at = time.time() + expires_in if expires_in else 0.0
        return token

    def invalidate(self) -> None:
        self._expires_at = 0.0
        self._access_token = ""


class _accounts_file_lock:
    def __init__(self, lock_path, timeout=15.0, delay=0.05):
        self.lock_path = lock_path
        self.timeout = timeout
        self.delay = delay
        self.identity = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4()}"

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(self.identity)
                except Exception:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    raise
                break
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    try:
                        os.remove(self.lock_path)
                    except OSError:
                        pass
                    continue
                time.sleep(self.delay)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if os.path.exists(self.lock_path):
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content == self.identity:
                    os.remove(self.lock_path)
        except OSError:
            pass


class ColabAccountStore:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path else Path(DEFAULT_ACCOUNTS_PATH)
        self._profiles: dict[str, AccountProfile] = self._load()

    def _load_raw(self) -> dict[str, AccountProfile]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        out: dict[str, AccountProfile] = {}
        for name, item in raw.get("profiles", raw).items():
            out[name] = AccountProfile(name=name, client_id=item.get("client_id", ""),
                client_secret=item.get("client_secret", ""), refresh_token=item.get("refresh_token", ""),
                access_token=item.get("access_token", ""), email=item.get("email", ""),
                scopes=item.get("scopes") or list(DEFAULT_SCOPES))
        return out

    def _load(self) -> dict[str, AccountProfile]:
        lock_path = self.path.with_name(self.path.name + ".lock")
        with _accounts_file_lock(lock_path):
            return self._load_raw()

    def _save_raw(self) -> None:
        payload = {"profiles": {
            name: {"client_id": p.client_id, "client_secret": p.client_secret, "refresh_token": p.refresh_token,
                "access_token": p.access_token, "email": p.email, "scopes": p.scopes} for name, p in
            sorted(self._profiles.items())}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(str(tmp_path), str(self.path))

    def save(self) -> None:
        lock_path = self.path.with_name(self.path.name + ".lock")
        with _accounts_file_lock(lock_path):
            self._save_raw()

    def update_profile_token(self, account_name: str, access_token: str) -> None:
        lock_path = self.path.with_name(self.path.name + ".lock")
        with _accounts_file_lock(lock_path):
            self._profiles = self._load_raw()
            if account_name in self._profiles:
                self._profiles[account_name].access_token = access_token
                self._save_raw()

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles)

    def get(self, name: str) -> AccountProfile:
        if name not in self._profiles:
            known = ", ".join(self.list_profiles()) or "(none)"
            raise ColabError(f"Unknown profile {name!r}. Known: {known}")
        return self._profiles[name]

    def add_profile(self, name: str, *, client_id: str = "", client_secret: str = "", refresh_token: str = "",
            access_token: str = "", email: str = "", scopes: list[str] | None = None,
            save: bool = True) -> AccountProfile:
        profile = AccountProfile(name=name, client_id=client_id, client_secret=client_secret,
            refresh_token=refresh_token, access_token=access_token, email=email,
            scopes=scopes or list(DEFAULT_SCOPES))
        self._profiles[name] = profile
        if save:
            self.save()
        return profile

    def client(self, name: str, **kwargs: Any) -> ColabClient:
        return ColabClient(profile=self.get(name), **kwargs)


class ColabClient:
    def __init__(self, *, profile: AccountProfile | None = None, account: str | None = None,
            accounts_path: str | os.PathLike[str] | None = None, access_token: str | None = None,
            refresh_token: str | None = None, client_id: str | None = None, client_secret: str | None = None,
            app_name: str = "Code", extension_version: str = EXTENSION_VERSION, verbose: bool = False):
        if account is not None and profile is None:
            profile = ColabAccountStore(accounts_path).get(account)
        if profile is not None:
            access_token = access_token or profile.access_token
            refresh_token = refresh_token or profile.refresh_token
            client_id = client_id or profile.client_id
            client_secret = client_secret or profile.client_secret
        self.profile = profile
        self.account = account or (profile.name if profile else "")
        self.accounts_path = accounts_path
        self.tokens = TokenProvider(access_token=access_token, refresh_token=refresh_token, client_id=client_id,
            client_secret=client_secret)
        self.app_name = app_name
        self.extension_version = extension_version
        self.verbose = verbose
        self._jobs: dict[str, ColabJob] = {}
        self.session = requests.Session()

    def auth_status(self) -> dict[str, Any]:
        try:
            token = self.tokens.get_access_token()
            resp = self.session.get("https://oauth2.googleapis.com/tokeninfo", params={"access_token": token},
                timeout=20)
            if not resp.ok:
                return {"ok": False, "status": "invalid", "error": resp.text[:300]}
            payload = resp.json()
            secs = int(payload.get("expires_in", 0))
            return {"ok": secs > 0, "status": "expired" if secs <= 0 else "near_expiry" if secs < 900 else "valid",
                "seconds_remaining": secs, "scopes": payload.get("scope", "").split(),
                "email": payload.get("email", ""), "account": self.account}
        except Exception as exc:
            return {"ok": False, "status": "error", "error": str(exc)[:300]}

    def list_assignments(self) -> list[dict[str, Any]]:
        return self._request_json("GET", f"{COLAB_GAPI_DOMAIN}/v1/assignments").get("assignments", [])

    def list_runtimes(self, *, refresh_connections: bool = True) -> list[RuntimeAssignment]:
        runtimes: list[RuntimeAssignment] = []
        for item in self.list_assignments():
            endpoint = item.get("endpoint")
            proxy = item.get("runtimeProxyInfo") or {}
            if refresh_connections and endpoint and not proxy.get("token"):
                try:
                    base_url, token = self._refresh_runtime_token(endpoint)
                    proxy = {"url": base_url, "token": token}
                except Exception:
                    proxy = {}
            if not endpoint:
                continue
            runtimes.append(
                RuntimeAssignment(assignment_id=str(uuid.uuid4()), endpoint=endpoint, base_url=proxy.get("url", ""),
                    runtime_token=proxy.get("token", ""), variant=_normalize_variant(item.get("variant")),
                    accelerator=str(item.get("accelerator") or "").upper(),
                    highmem=item.get("machineShape") in ("HIGHMEM", "SHAPE_HIGH_MEM", 1), raw=item))
        return runtimes

    def assign_runtime(self, *, variant: str = "DEFAULT", accelerator: str | None = None, highmem: bool = False,
            runtime_version: str | None = None, strict_accelerator: bool = True) -> RuntimeAssignment:
        assignment_id = str(uuid.uuid4())
        url = _build_assign_url(assignment_id, variant, accelerator, highmem, runtime_version)
        first = self._request_json("GET", url)
        if "runtimeProxyInfo" in first:
            return self._parse_assignment(assignment_id, first, requested=accelerator, strict=strict_accelerator,
                                          highmem=highmem)
        xsrf = first.get("token")
        if not xsrf:
            raise ColabError(f"Assignment GET returned neither runtimeProxyInfo nor XSRF token: {first}")
        second = self._request_json("POST", url, extra_headers={"X-Goog-Colab-Token": xsrf})
        _raise_for_outcome(second)
        return self._parse_assignment(assignment_id, second, requested=accelerator, strict=strict_accelerator,
                                      highmem=highmem)

    def unassign_runtime(self, runtime_or_endpoint: RuntimeAssignment | ColabRuntime | str) -> None:
        endpoint = _extract_endpoint(runtime_or_endpoint)
        url = f"{COLAB_DOMAIN}/tun/m/unassign/{endpoint}?authuser=0"
        try:
            payload = self._request_json("GET", url)
            xsrf = payload.get("token")
            if xsrf:
                self._request_empty("POST", url, extra_headers={"X-Goog-Colab-Token": xsrf})
        except ColabError:
            pass

    def runtime(self, *, variant: str = "DEFAULT", accelerator: str | None = None, highmem: bool = False,
            runtime_version: str | None = None, strict_accelerator: bool = True, auto_unassign: bool = True) -> ColabRuntime:
        assignment = self.assign_runtime(variant=variant, accelerator=accelerator, highmem=highmem,
            runtime_version=runtime_version, strict_accelerator=strict_accelerator)
        return ColabRuntime(self, assignment, auto_unassign=auto_unassign)

    def _control_headers(self) -> dict[str, str]:
        token = self.tokens.get_access_token()
        if self.profile and token != self.profile.access_token:
            if self.account:
                try:
                    store = ColabAccountStore(self.accounts_path)
                    store.update_profile_token(self.account, token)
                    self.profile.access_token = token
                except Exception:
                    pass
        return {"Authorization": f"Bearer {token}", "Accept": "application/json", "X-Colab-Client-Agent": "vscode",
            "X-Colab-VS-Code-App-Name": self.app_name, "X-Colab-VS-Code-Extension-Version": self.extension_version}

    def _request_json(self, method: str, url: str, *, extra_headers: dict[str, str] | None = None,
            json_body: dict[str, Any] | None = None, timeout: float = 120.0) -> Any:
        headers = {**self._control_headers(), **(extra_headers or {})}
        if self.verbose:
            print(f"[colab] {method} {url}")
        kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout}
        if json_body is not None:
            kwargs["json"] = json_body
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code == 401:
            self.tokens.invalidate()
            new_token = self.tokens.get_access_token()
            headers["Authorization"] = f"Bearer {new_token}"
            if self.account:
                try:
                    store = ColabAccountStore(self.accounts_path)
                    store.update_profile_token(self.account, new_token)
                    if self.profile:
                        self.profile.access_token = new_token
                except Exception:
                    pass
            resp = self.session.request(method, url, **kwargs)
        return _checked_json(resp)

    def _request_empty(self, method: str, url: str, *, extra_headers: dict[str, str] | None = None,
            timeout: float = 60.0) -> None:
        headers = {**self._control_headers(), **(extra_headers or {})}
        resp = self.session.request(method, url, headers=headers, timeout=timeout)
        if not resp.ok and resp.status_code not in (404,):
            raise ColabError(f"{method} {url} → {resp.status_code} {resp.reason}\n{resp.text[:500]}")

    def _refresh_runtime_token(self, endpoint: str) -> tuple[str, str]:
        params = urlencode({"endpoint": endpoint, "port": "8080"})
        payload = self._request_json("GET", f"{COLAB_GAPI_DOMAIN}/v1/runtime-proxy-token?{params}")
        url = payload.get("url")
        token = payload.get("token")
        if not url or not token:
            raise ColabError(f"Runtime token refresh missing url/token: {payload}")
        return url, token

    def _parse_assignment(self, assignment_id: str, payload: dict[str, Any], *, requested: str | None, strict: bool,
            highmem: bool) -> RuntimeAssignment:
        proxy = payload.get("runtimeProxyInfo") or {}
        endpoint = payload.get("endpoint")
        base_url = proxy.get("url")
        runtime_token = proxy.get("token")
        if not endpoint or not base_url or not runtime_token:
            raise ColabError(f"Assignment response missing endpoint/runtimeProxyInfo: {payload}")
        actual_accel = str(payload.get("accelerator") or "").upper()
        if requested and strict and actual_accel and actual_accel != requested.upper():
            raise ColabError(f"Requested accelerator {requested!r}, Colab assigned {actual_accel!r}.")
        assignment = RuntimeAssignment(assignment_id=assignment_id, endpoint=endpoint, base_url=base_url,
            runtime_token=runtime_token, variant=_normalize_variant(payload.get("variant")), accelerator=actual_accel,
            highmem=highmem or payload.get("machineShape") in ("HIGHMEM", "SHAPE_HIGH_MEM", 1), raw=payload)
        if self.verbose:
            print(f"[colab] assigned {assignment.endpoint} ({assignment.label})")
        return assignment


class ColabRuntime:
    def __init__(self, client: ColabClient, assignment: RuntimeAssignment, *, auto_unassign: bool = True):
        self.client = client
        self.assignment = assignment
        self.auto_unassign = auto_unassign
        self.closed = False
        self._jobs: dict[str, ColabJob] = {}
        self._keepalive_stop: threading.Event | None = None
        self._keepalive_thread: threading.Thread | None = None

    def close(self) -> None:
        self.stop_keepalive()
        if not self.closed and self.auto_unassign:
            try:
                self.client.unassign_runtime(self.assignment)
            except Exception:
                pass
        self.closed = True

    def keep_alive(self) -> None:
        self.client.keep_alive(self.assignment)

    def start_keepalive(self, interval: float = 300.0) -> None:
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        stop = threading.Event()
        self._keepalive_stop = stop

        def _loop() -> None:
            while not stop.wait(interval):
                try:
                    self.keep_alive()
                    if self.client.verbose:
                        print(f"[colab] keepalive → {self.assignment.endpoint}")
                except Exception as exc:
                    if self.client.verbose:
                        print(f"[colab] keepalive failed: {exc}")

        self._keepalive_thread = threading.Thread(target=_loop, daemon=True)
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        if self._keepalive_stop:
            self._keepalive_stop.set()
        self._keepalive_stop = None
        self._keepalive_thread = None

    def kernel(self, kernel_name: str = "python3") -> ColabKernel:
        return ColabKernel(self.assignment, kernel_name=kernel_name, verbose=self.client.verbose, client=self.client)

    def execute(self, code: str, *, timeout: float = 900.0,
            on_output: Callable[[dict[str, Any]], None] | None = None) -> CellExecution:
        with self.kernel() as k:
            return k.execute(code, timeout=timeout, on_output=on_output)

    def run_shell(self, cmd: str, *, timeout: float = 120.0) -> CellExecution:
        return self.execute(f"import subprocess, sys\n"
                            f"_r = subprocess.run({cmd!r}, shell=True, capture_output=True, text=True)\n"
                            f"print(_r.stdout, end='')\n"
                            f"if _r.stderr: print(_r.stderr, end='', file=sys.stderr)\n"
                            f"if _r.returncode != 0: raise SystemExit(_r.returncode)", timeout=timeout)

    def start_job(self, code: str, *, name: str = "", timeout: float | None = None,
            on_output: Callable[[dict[str, Any]], None] | None = None) -> ColabJob:
        kernel = self.kernel()
        kernel.start()
        job = kernel.start_job(code, name=name, timeout=timeout, on_output=on_output)
        job._runtime = self
        self._jobs[job.job_id] = job
        self.client._jobs[job.job_id] = job
        return job


class ColabKernel:
    def __init__(self, assignment: RuntimeAssignment, *, kernel_name: str = "python3", verbose: bool = False,
            client: ColabClient | None = None):
        self.assignment = assignment
        self.kernel_name = kernel_name
        self.verbose = verbose
        self.client = client
        self.kernel_id: str | None = None
        self.session_id: str = str(uuid.uuid4())
        self.ws: Any = None

    def __enter__(self) -> ColabKernel:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self) -> None:
        _require_websocket()
        url = _runtime_url(self.assignment.base_url, "/api/kernels")
        headers = {**_runtime_headers(self.assignment.runtime_token), "Content-Type": "application/json"}
        data = json.dumps({"name": self.kernel_name})
        if self.client is not None:
            resp = self.client.session.post(url, headers=headers, data=data, timeout=60)
        else:
            resp = requests.post(url, headers=headers, data=data, timeout=60)
        payload = _checked_json(resp)
        self.kernel_id = payload.get("id")
        if not self.kernel_id:
            raise ColabError(f"Kernel start missing id: {payload}")
        ws_url = _websocket_url(self.assignment.base_url, self.kernel_id, self.session_id)
        for attempt in range(1, 4):
            try:
                self.ws = _websocket_module.create_connection(ws_url,
                    header=[f"{k}: {v}" for k, v in _runtime_headers(self.assignment.runtime_token).items()], timeout=30,
                    ping_interval=25, ping_timeout=10)
                break
            except Exception as exc:
                if attempt == 3:
                    raise ColabError(f"WebSocket connection to {ws_url} failed after 3 attempts: {exc}")
                time.sleep(3.0)
        if self.verbose:
            print(f"[colab] kernel {self.kernel_id} started")

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self.kernel_id:
            try:
                url = _runtime_url(self.assignment.base_url, f"/api/kernels/{self.kernel_id}")
                headers = _runtime_headers(self.assignment.runtime_token)
                if self.client is not None:
                    self.client.session.delete(url, headers=headers, timeout=20)
                else:
                    requests.delete(url, headers=headers, timeout=20)
            except Exception:
                pass
            if self.verbose:
                print(f"[colab] kernel {self.kernel_id} stopped")
            self.kernel_id = None

    def execute(self, code: str, *, index: int = 1, timeout: float = 900.0,
            on_output: Callable[[dict[str, Any]], None] | None = None) -> CellExecution:
        if self.ws is None:
            raise ColabError("Kernel not started. Use 'with runtime.kernel() as k:'.")
        msg_id, msg = _jupyter_message("execute_request", self.session_id,
            {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False,
                "stop_on_error": True})
        self.ws.send(json.dumps(msg))
        return _collect_execution(self.ws, msg_id, index, code, timeout, on_output=on_output)

    def start_job(self, code: str, *, name: str = "", index: int = 1, timeout: float | None = None,
            on_output: Callable[[dict[str, Any]], None] | None = None) -> ColabJob:
        if self.ws is None:
            raise ColabError("Kernel not started.")
        msg_id, msg = _jupyter_message("execute_request", self.session_id,
            {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False,
                "stop_on_error": True})
        self.ws.send(json.dumps(msg))
        job = ColabJob(job_id=msg_id, name=name or msg_id[:12], kernel=self, code=code, index=index, timeout=timeout,
            on_output=on_output)
        job.start()
        return job


class ColabJob:
    def __init__(self, *, job_id: str, name: str, kernel: ColabKernel, code: str, index: int, timeout: float | None,
            on_output: Callable[[dict[str, Any]], None] | None = None):
        self.job_id = job_id
        self.name = name
        self.kernel = kernel
        self.code = code
        self.index = index
        self.timeout = timeout
        self.on_output = on_output
        self.status: str = "queued"
        self.started_at: float = time.time()
        self.finished_at: float | None = None
        self.result = CellExecution(index=index, code=code)
        self.error: str = ""
        self._thread: threading.Thread | None = None
        self._runtime: ColabRuntime | None = None

    def tail(self, n: int = 20) -> str:
        return self.result.tail(n)

    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            if self.kernel.ws is None:
                raise ColabError("Job kernel WebSocket is closed.")
            self.result = _collect_execution(self.kernel.ws, self.job_id, self.index, self.code,
                self.timeout if self.timeout is not None else 10 ** 9, on_output=self._handle_output,
                result=self.result)
            if self.status != "cancelled":
                self.status = "failed" if self.result.errored else "succeeded"
        except TimeoutError:
            if self.status != "cancelled":
                self.status = "running"
        except Exception as exc:
            self.error = str(exc)
            if self.status != "cancelled":
                self.status = "failed"
        finally:
            if self.status != "running":
                self.finished_at = time.time()
                self.result.finished_at = self.finished_at

    def _handle_output(self, output: dict[str, Any]) -> None:
        if self.on_output:
            try:
                self.on_output(output)
            except Exception:
                pass

    def watcher(self) -> ColabWatcher:
        return ColabWatcher(self)


class ColabWatcher:
    def __init__(self, job: ColabJob):
        self.job = job

    def wait_for_pattern(self, pattern: str, *, timeout: float = 60.0, group: int = 0, flags: int = 0) -> str | None:
        rx = re.compile(pattern, flags)
        seen = 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            for out in self.job.result.outputs[seen:]:
                text = _output_text(out)
                for line in text.splitlines():
                    m = rx.search(line)
                    if m:
                        return m.group(group)
            seen = len(self.job.result.outputs)
            if self.job.status not in ("running", "queued"):
                break
            time.sleep(0.3)
        return None


def _checked_json(response: requests.Response) -> Any:
    if not response.ok:
        raise ColabError(f"{response.request.method} {response.url} → "
                         f"{response.status_code} {response.reason}\n{response.text[:1000]}")
    text = response.text
    if text.startswith(JSON_PREFIX):
        text = text[len(JSON_PREFIX):]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ColabError(f"Non-JSON from {response.url}: {text[:500]}") from exc


def _require_websocket() -> None:
    if _websocket_module is None:
        raise ColabError("Missing dependency: pip install websocket-client") from _WEBSOCKET_IMPORT_ERROR


def _raise_for_outcome(payload: dict[str, Any]) -> None:
    outcome = payload.get("outcome")
    if outcome in _OUTCOME_QUOTA:
        raise ColabQuotaError("Colab reported insufficient quota for this assignment.")
    if outcome in _OUTCOME_BLOCKED:
        raise ColabBlockedError("Colab reported this account is blocked from runtimes.")


def _build_assign_url(assignment_id: str, variant: str, accelerator: str | None, highmem: bool,
        runtime_version: str | None) -> str:
    params: dict[str, str] = {"nbh": _nbh(assignment_id), "authuser": "0"}
    if variant != "DEFAULT":
        params["variant"] = variant
    if accelerator:
        params["accelerator"] = accelerator
    if highmem:
        params["shape"] = "hm"
    if runtime_version:
        params["runtime_version_label"] = runtime_version
    return f"{COLAB_DOMAIN}/tun/m/assign?{urlencode(params)}"


def _nbh(value: str) -> str:
    compact = value.replace("-", "_")
    return compact + "." * max(0, 44 - len(compact))


def _normalize_variant(value: Any) -> str:
    if value in (1, "GPU", "VARIANT_GPU"):
        return "GPU"
    if value in (2, "TPU", "VARIANT_TPU"):
        return "TPU"
    return "DEFAULT"


def _runtime_headers(runtime_token: str) -> dict[str, str]:
    return {"X-Colab-Runtime-Proxy-Token": runtime_token, "X-Colab-Client-Agent": "vscode"}


def _runtime_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _websocket_url(base_url: str, kernel_id: str, session_id: str) -> str:
    parsed = urlparse(_runtime_url(base_url, f"/api/kernels/{kernel_id}/channels?session_id={session_id}"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _jupyter_message(msg_type: str, session_id: str, content: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    msg_id = uuid.uuid4().hex
    return msg_id, {"header": {"msg_id": msg_id, "username": "username", "session": session_id, "msg_type": msg_type,
        "version": "5.3", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "parent_header": {},
        "metadata": {}, "content": content, "channel": "shell"}


def _collect_execution(ws: Any, msg_id: str, index: int, code: str, timeout: float, *,
        on_output: Callable[[dict[str, Any]], None] | None = None,
        result: CellExecution | None = None) -> CellExecution:
    import socket
    deadline = time.time() + timeout
    result = result or CellExecution(index=index, code=code)
    shell_done = False
    iopub_idle = False

    while time.time() < deadline:
        ws.settimeout(max(0.1, min(5.0, deadline - time.time())))
        try:
            raw = ws.recv()
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            is_timeout = (
                    isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in exc_name or "timed out" in str(
                exc).lower())
            if is_timeout:
                continue
            raise ColabRuntimeDeadError(f"WebSocket closed unexpectedly: {exc}") from exc

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        message = json.loads(raw)
        parent_id = message.get("parent_header", {}).get("msg_id")
        channel = message.get("channel")
        msg_type = message.get("header", {}).get("msg_type")
        content = message.get("content", {})

        if msg_type == "colab_request":
            continue
        if parent_id != msg_id:
            continue

        if msg_type == "execute_input":
            result.execution_count = content.get("execution_count")
        output = _output_from_iopub(message)
        if output is not None:
            result.outputs.append(output)
            if on_output:
                try:
                    on_output(output)
                except Exception:
                    pass
            if output.get("output_type") == "error":
                result.errored = True

        if channel == "shell" and msg_type == "execute_reply":
            shell_done = True
            result.errored = result.errored or content.get("status") == "error"
        if channel == "iopub" and msg_type == "status" and content.get("execution_state") == "idle":
            iopub_idle = True
        if shell_done and iopub_idle:
            result.finished_at = time.time()
            return result

    raise TimeoutError(f"Cell {index} timed out after {timeout:.0f}s.")


def _output_from_iopub(message: dict[str, Any]) -> dict[str, Any] | None:
    msg_type = message.get("header", {}).get("msg_type")
    content = message.get("content", {})
    if msg_type == "stream":
        return {"output_type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")}
    if msg_type in ("display_data", "execute_result"):
        out: dict[str, Any] = {"output_type": msg_type, "data": content.get("data", {}),
            "metadata": content.get("metadata", {})}
        if msg_type == "execute_result":
            out["execution_count"] = content.get("execution_count")
        return out
    if msg_type == "error":
        return {"output_type": "error", "ename": content.get("ename", ""), "evalue": content.get("evalue", ""),
            "traceback": content.get("traceback", [])}
    return None


def _cell_text(outputs: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for out in outputs:
        chunks.append(_output_text(out))
    return "".join(chunks)


def _output_text(output: dict[str, Any]) -> str:
    ot = output.get("output_type")
    if ot == "stream":
        return str(output.get("text", ""))
    if ot == "error":
        tb = output.get("traceback") or []
        return "\n".join(map(str, tb)) or str(output.get("evalue", ""))
    data = output.get("data") or {}
    value = data.get("text/plain", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _extract_endpoint(value: RuntimeAssignment | ColabRuntime | str) -> str:
    if isinstance(value, ColabRuntime):
        return value.assignment.endpoint
    if isinstance(value, RuntimeAssignment):
        return value.endpoint
    return str(value)
