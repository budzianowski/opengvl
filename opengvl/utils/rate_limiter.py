import threading
import time
from contextlib import ContextDecorator

SECS_PER_MIN = 60.0


class RateLimiter(ContextDecorator):
    """Simple rate limiter context manager.

    Limits the rate of entry to at most `max_calls` per `period` seconds.
    Uses a thread-safe token bucket implemented via sleep and a rolling window.
    Designed for coarse-grained RPM limiting around API calls.
    """

    def __init__(self, *, max_calls: float, period: float) -> None:
        if max_calls <= 0 or period <= 0:
            raise ValueError
        self._lock = threading.Lock()
        self.max_calls = float(max_calls)
        self.period = float(period)
        self._calls: list[float] = []  # timestamps of recent calls

    def __enter__(self):
        now = time.monotonic()
        with self._lock:
            # Drop timestamps older than period
            cutoff = now - self.period
            self._calls = [t for t in self._calls if t > cutoff]
            if len(self._calls) >= self.max_calls:
                # Sleep until the oldest call exits the window
                sleep_for = self._calls[0] + self.period - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                # Recompute window after sleep
                now = time.monotonic()
                cutoff = now - self.period
                self._calls = [t for t in self._calls if t > cutoff]
            # Record this call timestamp
            self._calls.append(now)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Nothing special to do; timestamp already recorded in __enter__
        return False
