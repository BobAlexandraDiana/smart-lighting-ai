from ANN.ANN import ANN
import numpy as np
import sys


def predict(email, hours, sunset, sunrise, eyeDiseases, activity):
    normalizedEntry = normalizeEntry(hours, sunset, sunrise, eyeDiseases, activity)
    ann = ANN(email)
    print(ann.predict(normalizedEntry))

def normalizeEntry(hours, sunset, sunrise, eyeDiseases, activity):
    nonNormalizedEntries = []
    nonNormalizedEntries.append(hours)
    nonNormalizedEntries.append(sunset)
    nonNormalizedEntries.append(sunrise)
    nonNormalizedEntries.append(eyeDiseases)
    nonNormalizedEntries.append(activity)

    return constructNormalizedEntryList(nonNormalizedEntries)

def constructNormalizedEntryList(nonNormalizedEntries):
    currentEntry = []

    normalizedHours = int(nonNormalizedEntries[0]) / 24.0
    currentEntry.append(normalizedHours)

    normalizedSunset = int(nonNormalizedEntries[1]) / 12.0
    currentEntry.append(normalizedSunset)

    normalizedSunrise = int(nonNormalizedEntries[2]) / 12.0
    currentEntry.append(normalizedSunrise)

    normalizedEyeDiseases = float(nonNormalizedEntries[3])
    currentEntry.append(normalizedEyeDiseases)

    if (nonNormalizedEntries[4] == "study"):
        activities = [1, 0, 0, 0, 0, 0, 0, 0]
    if (nonNormalizedEntries[4] == "read"):
        activities = [0, 1, 0, 0, 0, 0, 0, 0]
    if (nonNormalizedEntries[4] == "rest"):
        activities = [0, 0, 1, 0, 0, 0, 0, 0]
    if (nonNormalizedEntries[4] == "sleep"):
        activities = [0, 0, 0, 1, 0, 0, 0, 0]
    if (nonNormalizedEntries[4] == "laptop/TV"):
        activities = [0, 0, 0, 0, 1, 0, 0, 0]
    if (nonNormalizedEntries[4] == "sport"):
        activities = [0, 0, 0, 0, 0, 1, 0, 0]
    if (nonNormalizedEntries[4] == "house-activities"):
        activities = [0, 0, 0, 0, 0, 0, 1, 0]
    if (nonNormalizedEntries[4] == "friends-night-at-home"):
        activities = [0, 0, 0, 0, 0, 0, 0, 1]
    currentEntry += activities

    return np.array(currentEntry)

if __name__ == "__main__":
    # predict("bobalexandradiana@gmail.com", 22,8,4,1,"study")
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
