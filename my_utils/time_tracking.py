import time
from dataclasses import dataclass

class baseTracker: 
    name = "default"
    start_time = 0
    stop_time = 0
    total_time = 0

    def __init__(self, name = None):
        self.start_time = time.perf_counter()
        if name:
            self.name = name

    def stop(self):
        self.stop_time = time.perf_counter()
        self.total_time = self.stop_time - self.start_time 
        return self.total_time
