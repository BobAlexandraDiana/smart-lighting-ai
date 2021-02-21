import sys
import os

def retrain(email, hours, sunset, sunrise, eyeDiseases, activityType, temperature):
    scriptToRun = "python3 StartAppRetrain.py" + " " + email + " " + hours + " " + sunset + " " + sunrise + " " + eyeDiseases + " " + activityType + " " + temperature

    os.system(scriptToRun)


if __name__ == "__main__":
    retrain(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
