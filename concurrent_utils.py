import fcntl
import os
from pathlib import Path

from config import LOCKS_PATH


def acquire_lock(service: str) -> tuple[Path, object]:
    LOCKS_PATH.mkdir(parents=True, exist_ok=True)
    lock_path = LOCKS_PATH / f"{service}.lock"
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_path, lock_file