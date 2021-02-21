import numpy as np
import random


class NormalizedDataset:
    def __init__(self, filename):
        self._sample_matrix = []
        try:
            with open(filename, 'r') as fileDescriptor:
                lines = fileDescriptor.readlines()

                self._verdicts_vector = np.zeros(len(lines))
                index = 0

                for currentLine in lines:
                    normalizedEntry = self.constructNormalizedEntryList(currentLine.split(" "))
                    self._sample_matrix.append(normalizedEntry)

                    self._verdicts_vector[index] = int(currentLine.split(" ")[5].rstrip("\n")) / 10000.0
                    index += 1

            self._sample_matrix = np.matrix(self._sample_matrix)
        except FileNotFoundError as ex:
            print("Error: File not found! Details: " + str(ex))

    def getSampleMatrix(self):
        return self._sample_matrix

    def getVerdictsVector(self):
        return self._verdicts_vector

    def constructNormalizedEntryList(self, listOfTokens):
        currentEntry = []

        normalizedHours = int(listOfTokens[0]) / 24.0
        currentEntry.append(normalizedHours)

        normalizedSunset = int(listOfTokens[1]) / 12.0
        currentEntry.append(normalizedSunset)

        normalizedSunrise = int(listOfTokens[2]) / 12.0
        currentEntry.append(normalizedSunrise)

        normalizedEyeDiseases = float(listOfTokens[3])
        currentEntry.append(normalizedEyeDiseases)

        if (listOfTokens[4] == "study"):
            activities = [1, 0, 0, 0, 0, 0, 0, 0]
        if (listOfTokens[4] == "read"):
            activities = [0, 1, 0, 0, 0, 0, 0, 0]
        if (listOfTokens[4] == "rest"):
            activities = [0, 0, 1, 0, 0, 0, 0, 0]
        if (listOfTokens[4] == "sleep"):
            activities = [0, 0, 0, 1, 0, 0, 0, 0]
        if (listOfTokens[4] == "laptop/TV"):
            activities = [0, 0, 0, 0, 1, 0, 0, 0]
        if (listOfTokens[4] == "sport"):
            activities = [0, 0, 0, 0, 0, 1, 0, 0]
        if (listOfTokens[4] == "house-activities"):
            activities = [0, 0, 0, 0, 0, 0, 1, 0]
        if (listOfTokens[4] == "friends-night-at-home"):
            activities = [0, 0, 0, 0, 0, 0, 0, 1]
        currentEntry += activities

        return np.array(currentEntry)

    def shuffleDataset(self):
        indices= np.arange(self._sample_matrix.shape[0])
        random.shuffle(indices)

        self._sample_matrix = self._sample_matrix[indices]
        self._verdicts_vector = self._verdicts_vector[indices]

    def getTrainingDataSamples(self):
        return self._sample_matrix[:int(0.8 * self._sample_matrix.shape[0])]

    def getTrainingDataVerdicts(self):
        return self._verdicts_vector[:int(0.8 * self._verdicts_vector.shape[0])]

    def getValidationDataSamples(self):
        return self._sample_matrix[int(0.8 * self._sample_matrix.shape[0]):]

    def getValidationDataVerdicts(self):
        return self._verdicts_vector[int(0.8 * self._verdicts_vector.shape[0]):]

