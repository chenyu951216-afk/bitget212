import json
import os
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional

_IO_LOCK = threading.RLock()


def atomic_json_save(path: str, data: Any, *, ensure_ascii: bool = False, indent: int = 2) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _IO_LOCK:
        fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.json', dir=directory or None)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, ensure_ascii=ensure_ascii, indent=indent)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def atomic_json_load(path: str, default: Any) -> Any:
    try:
        with _IO_LOCK:
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    except Exception:
        return default


def safe_request_json(requests_module, method: str, url: str, *, timeout: float = 8, retries: int = 2,
                      logger: Optional[Callable[[str], None]] = None, **kwargs):
    last_error = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = requests_module.request(method=method.upper(), url=url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            try:
                return resp.json(), None
            except Exception:
                return None, f'非JSON回應: {resp.text[:200]}'
        except Exception as exc:
            last_error = str(exc)
            if logger:
                logger(f'HTTP {method.upper()} 失敗 attempt {attempt}: {url} | {last_error[:240]}')
            if attempt < retries:
                time.sleep(min(1.2 * attempt, 3.0))
    return None, last_error or 'unknown error'


def safe_request_text(requests_module, method: str, url: str, *, timeout: float = 8, retries: int = 2,
                      logger: Optional[Callable[[str], None]] = None, **kwargs):
    last_error = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = requests_module.request(method=method.upper(), url=url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp.text, None
        except Exception as exc:
            last_error = str(exc)
            if logger:
                logger(f'HTTP {method.upper()} 失敗 attempt {attempt}: {url} | {last_error[:240]}')
            if attempt < retries:
                time.sleep(min(1.2 * attempt, 3.0))
    return None, last_error or 'unknown error'


def prune_mapping(mapping: Dict[str, Any], *, max_size: int = 500, prune_count: int = 200) -> None:
    if len(mapping) <= max_size:
        return
    for key in list(mapping.keys())[:prune_count]:
        mapping.pop(key, None)


def snapshot_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    return dict(mapping or {})
