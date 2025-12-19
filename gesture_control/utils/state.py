import time
from collections import deque

class Cooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.t = 0.0

    def ready(self):
        return (time.time() - self.t) >= self.seconds

    def hit(self):
        self.t = time.time()

class SwipeDetector:
    def __init__(self, maxlen=12, dx_thresh=0.55, cooldown_s=0.6):
        self.x_hist = deque(maxlen=maxlen)
        self.dx_thresh = dx_thresh
        self.cooldown = Cooldown(cooldown_s)

    def update(self, wrist_x: float):
        self.x_hist.append(wrist_x)

    def detect(self):
        if not self.cooldown.ready():
            return None
        if len(self.x_hist) < self.x_hist.maxlen:
            return None
        dx = self.x_hist[-1] - self.x_hist[0]
        if abs(dx) >= self.dx_thresh:
            self.cooldown.hit()
            self.x_hist.clear()
            return "swipe_right" if dx > 0 else "swipe_left"
        return None
