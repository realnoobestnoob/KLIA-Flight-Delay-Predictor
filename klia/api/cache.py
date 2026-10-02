"""Model cache: the newest bundle lives in memory (and a copy on disk). Requests never trigger training."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from klia.model.bundle import Bundle

log = logging.getLogger("klia.cache")


class ModelCache:
    def __init__(self, store, cache_dir: Path, refresh_seconds: int):
        self.store = store
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.refresh_seconds = refresh_seconds
        self.bundle: Bundle | None = None
        self.version: int | None = None
        self.loaded_at = 0.0
        self.checked_at = 0.0
        self._busy = threading.Lock()

    # -- loading -----------------------------------------------------------
    def _load_from_disk(self) -> bool:
        files = sorted(self.dir.glob("bundle_*.gz"), key=lambda p: int(p.stem.split("_")[1]))
        for f in reversed(files):
            try:
                self.bundle, self.version = Bundle.loads(f.read_bytes()), int(f.stem.split("_")[1])
                self.loaded_at = time.time()
                log.info("loaded v%s from local cache", self.version)
                return True
            except Exception as e:                       # corrupt file: try the next one
                log.warning("bad cache file %s: %s", f, e)
        return False

    def _save_to_disk(self, version: int, blob: bytes) -> None:
        tmp = self.dir / f".tmp_{version}"
        tmp.write_bytes(blob)
        os.replace(tmp, self.dir / f"bundle_{version}.gz")
        for old in sorted(self.dir.glob("bundle_*.gz"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
            old.unlink(missing_ok=True)

    def refresh(self) -> bool:
        """Pull the newest bundle if the registry has a newer one. Returns True if the model changed."""
        self.checked_at = time.monotonic()
        latest = self.store.latest_version()
        if latest is None or latest == self.version:
            return False
        got = self.store.load_bundle_bytes(latest)
        if got is None:
            return False
        version, blob = got
        new = Bundle.loads(blob)                         # parse first; swap only if it loads
        self._save_to_disk(version, blob)
        self.bundle, self.version, self.loaded_at = new, version, time.time()
        log.info("loaded model v%s (%d rows)", version, new.model.n_learned)
        return True

    def startup(self) -> None:
        try:
            self.refresh()
        except Exception as e:
            log.warning("registry unreachable at startup (%s); trying local cache", e)
        if self.bundle is None:
            self._load_from_disk()

    # -- serving -----------------------------------------------------------
    def get(self) -> Bundle | None:
        """Return the cached bundle instantly; if it is time, check for a newer one in the background."""
        if time.monotonic() - self.checked_at > self.refresh_seconds and self._busy.acquire(blocking=False):
            def work():
                try:
                    self.refresh()
                except Exception as e:
                    log.warning("refresh failed, keeping v%s: %s", self.version, e)
                finally:
                    self.checked_at = time.monotonic()
                    self._busy.release()
            threading.Thread(target=work, daemon=True).start()
        return self.bundle
