import time

class Chronometer:
    def start(self):
        self.t0 = time.time()

    def stop(self):
        return time.time() - self.t0