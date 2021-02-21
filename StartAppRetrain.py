import sys
import os
import signal

from ANN.ANN import ANN


def handler(signum, frame):
    print("NO!")


def retrain(email, hours, sunset, sunrise, eyeDiseases, activityType, temperature):
    ann = ANN(email)
    ann.retrain(email, hours, sunset, sunrise, eyeDiseases, activityType, int(temperature))
    print("Retrain performed successfully!")


if __name__ == "__main__":
    os.setpgrp()
    signal.signal(signal.SIGTERM, handler)
    retrain(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
