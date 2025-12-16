import time
from typing import Callable, Optional


class Progress:

    def __init__(
        self,
        callback: Optional[Callable[[float, str], None]] = None,
        enable_time: bool = True,
        min_interval: float = 0.05, 
        min_step: float = 0.2,
    ):
        self.callback = callback
        self.enable_time = enable_time
        self.min_interval = float(min_interval)
        self.min_step = float(min_step)

        self.start_time = time.time()
        self._last_time = 0.0
        self._last_percent = -1.0
        self._last_message = ""

    def update(self, percent: float, message: str = "", force: bool = False):
        percent = max(0.0, min(100.0, float(percent)))
        now = time.time()

        if not force:
            too_soon = (now - self._last_time) < self.min_interval
            too_small = abs(percent - self._last_percent) < self.min_step
            same_msg = (message == self._last_message)

            if (too_soon and too_small and same_msg):
                return

        self._last_time = now
        self._last_percent = percent
        self._last_message = message

        if self.callback:
            self.callback(percent, message)
        else:
            if message:
                print(f"[{percent:6.2f}%] {message}")
            else:
                print(f"[{percent:6.2f}%]")

    def stage(self, percent: float, message: str):
        self.update(percent, message, force=True)

    def done(self, message: str = "Hoàn tất"):
        elapsed = time.time() - self.start_time
        if self.enable_time:
            message = f"{message} (thời gian: {elapsed:.2f}s)"
        self.update(100.0, message, force=True)

def console_progress(percent: float, message: str = ""):
    """Callback đơn giản cho console."""
    if message:
        print(f"[{percent:6.2f}%] {message}")
    else:
        print(f"[{percent:6.2f}%]")


def silent_progress(percent: float, message: str = ""):
    """Callback rỗng (không làm gì)."""
    return
