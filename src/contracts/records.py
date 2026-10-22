"""Canonical, immutable runtime artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path


WINDOWS_SHARING_VIOLATIONS = frozenset({5, 32, 33})
ATOMIC_WRITE_ATTEMPTS = 6
_publication_locks_guard = threading.Lock()
_publication_locks = {}


def _publication_lock(path):
    key = str(Path(path).resolve()).casefold()
    with _publication_locks_guard:
        return _publication_locks.setdefault(key, threading.RLock())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, *, attempts=ATOMIC_WRITE_ATTEMPTS):
    """Read one complete snapshot, tolerating transient Windows publication races."""

    target = Path(path)
    for attempt in range(attempts):
        try:
            return json.loads(target.read_text(encoding='utf-8-sig'))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
            winerror = getattr(exc, 'winerror', None)
            retryable = (
                isinstance(exc, (FileNotFoundError, PermissionError, json.JSONDecodeError))
                or winerror in WINDOWS_SHARING_VIOLATIONS
            )
            if not retryable or attempt + 1 == attempts:
                raise
            time.sleep(.005 * 2 ** attempt)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def atomic_write(path, text, *, immutable=False):
    with _publication_lock(path):
        return _atomic_write_unlocked(path, text, immutable=immutable)


def _atomic_write_unlocked(path, text, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode('utf-8')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                # Atomic create-if-absent, including simultaneous writers.
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError('immutable artifact conflict: ' + str(path))
        else:
            # Windows readers can briefly deny replacement while the scheduler
            # samples stages.json. Retry only the atomic rename, never a sender.
            for attempt in range(ATOMIC_WRITE_ATTEMPTS):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError as exc:
                    if getattr(exc, 'winerror', None) not in WINDOWS_SHARING_VIOLATIONS or attempt + 1 == ATOMIC_WRITE_ATTEMPTS:
                        # Preserve the native exception class while attaching
                        # structured facts consumed by the failure taxonomy.
                        exc.operation = 'atomic_replace'
                        exc.source_path_length = len(str(Path(temporary).resolve()))
                        exc.target_path_length = len(str(path.resolve()))
                        raise
                    time.sleep(.01 * 2 ** attempt)
        if os.name != 'nt':
            # Persist the directory entry as well as file contents where the
            # platform exposes a directory fsync.
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value, *, immutable=False):
    atomic_write(path, canonical(value) + '\n', immutable=immutable)


def write_jsonl(path, values, *, immutable=False):
    atomic_write(path, ''.join(canonical(v) + '\n' for v in values), immutable=immutable)


def snapshot(directory, kind, payload, parent=None):
    value = {'kind': kind, 'parent_hash': parent, 'payload': payload}
    key = digest(value)
    write_json(Path(directory) / (key + '.json'), value, immutable=True)
    return key


def seal(directory, filenames):
    root = Path(directory).resolve()
    files = {}
    for name in filenames:
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError('seal path escapes run directory')
        files[name] = file_hash(path)
    value = {'files': files, 'seal_hash': digest(files)}
    write_json(root / 'seal.json', value, immutable=True)
    return value


def verify_seal(directory):
    root = Path(directory).resolve()
    value = read_json(root / 'seal.json')
    if digest(value['files']) != value['seal_hash']:
        raise ValueError('seal manifest changed')
    for name, expected in value['files'].items():
        path = (root / name).resolve()
        if root not in path.parents or file_hash(path) != expected:
            raise ValueError('sealed artifact changed: ' + name)
    return value
