import time


class TapDetector:

    def __init__(self, threshold=0.05):

        self.threshold = threshold

        self.state = "OPEN"

        self.tap_count = 0

        self.tap_times = []

    def update(self, distance):

        if distance < self.threshold:

            if self.state == "OPEN":

                self.state = "CLOSED"

                self.tap_count += 1

                self.tap_times.append(time.time())

        else:

            self.state = "OPEN"

    def get_tap_count(self):

        return self.tap_count

    def get_intervals(self):

        if len(self.tap_times) < 2:
            return []

        intervals = []

        for i in range(1, len(self.tap_times)):

            intervals.append(
                self.tap_times[i] -
                self.tap_times[i - 1]
            )

        return intervals