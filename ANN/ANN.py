import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py


class ANN:
    def __init__(self, email):
        self._email = email
        self._path = "/Users/dianabob/Documents/LICENTA/Aplicatie/ANNSmartLighting"
        if self.readConfigFromFile(email):
            self._model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
            return

        self._model = keras.Sequential([
            keras.layers.Dense(20, activation=tf.nn.relu),
            keras.layers.Dense(20, activation=tf.nn.relu),
            keras.layers.Dense(20, activation=tf.nn.relu),
            keras.layers.Dense(20, activation=tf.nn.relu),
            keras.layers.Dense(20, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
        self._model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    def train(self, samplesMatrix, verdictsVector):
        early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=100)

        history = self._model.fit(samplesMatrix, verdictsVector, epochs=500, verbose=1, callbacks=[early_stop])

        return history

    def validate(self, sampleMatrix, verdictsVector):
        print(str(self._model.evaluate(sampleMatrix, verdictsVector)))

    def predict(self, inputEntry):
        resultList = self._model.predict(inputEntry.reshape((1, inputEntry.shape[0])))
        result = resultList[0]
        return result[0] * 10000

    def retrain(self, email, hours, sunset, sunrise, eyeDiseases, activityType, temperature):

        if os.path.exists(self._path + '/datasets/dataset' + "-" + email + '.txt'):
            filename = self._path + '/datasets/dataset' + "-" + email + '.txt'
        else:
            filename = self._path + '/datasets/initialDataset.txt'

        lines = []
        try:
            with open(filename, 'r') as fileDescriptor:
                lines = fileDescriptor.readlines()

        except FileNotFoundError as ex:
            print("Error: File not found for read! Details: " + str(ex))

        self.constructDatasetFileForUser(lines, email, hours, sunset, sunrise, eyeDiseases, activityType, temperature)

        sample_matrix = []
        try:
            with open(self._path + '/datasets/dataset' + "-" + email + '.txt', 'r') as fileDescriptor:
                lines = fileDescriptor.readlines()

                verdicts_vector = np.zeros(len(lines))
                index = 0

                for currentLine in lines:

                    normalizedEntry = self.constructNormalizedEntryList(currentLine.split(" "))
                    sample_matrix.append(normalizedEntry)

                    verdicts_vector[index] = int(currentLine.split(" ")[5].rstrip("\n")) / 10000.0
                    index += 1

            sample_matrix = np.matrix(sample_matrix)
        except FileNotFoundError as ex:
            print("Error: File not found! Details: " + str(ex))

        early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=100)
        self._model.fit(sample_matrix, verdicts_vector, epochs=30, verbose=1,callbacks=[early_stop])

        self.writeConfigToFile(email)

    def constructDatasetFileForUser(self, lines, email, hours, sunset, sunrise, eyeDiseases, activityType, temperature):
        linesToWriteToDatasetFile = []
        found = False
        for currentLine in lines:
            currentLineList = currentLine.split(" ")

            if int(currentLineList[0]) == hours and \
                    int(currentLineList[1]) == sunset and \
                    int(currentLineList[2]) == sunrise and \
                    int(currentLineList[3]) == eyeDiseases and \
                    currentLineList[4] == activityType:
                lineToWrite = currentLineList[0] + " " + currentLineList[1] + " " + currentLineList[2] + " " + \
                              currentLineList[3] + " " + currentLineList[4] + " " + str(temperature)
                found = True
            else:
                lineToWrite = currentLineList[0] + " " + currentLineList[1] + " " + currentLineList[2] + " " + \
                              currentLineList[3] + " " + currentLineList[4] + " " + currentLineList[5].rstrip("\n")

            linesToWriteToDatasetFile.append(lineToWrite)

        if (found == False):
            newEntry = str(hours) + " " + str(sunset) + " " + str(sunrise) + " " + str(
                eyeDiseases) + " " + activityType + " " + str(temperature)
            linesToWriteToDatasetFile.append(newEntry)

        try:
            with open(self._path + '/datasets/dataset' + "-" + email + '.txt', 'w') as fileDescriptor:
                contor = 0
                for currentLine in linesToWriteToDatasetFile:
                    fileDescriptor.write(currentLine)
                    contor += 1
                    if (contor < len(linesToWriteToDatasetFile)):
                        fileDescriptor.write("\n")

        except FileNotFoundError as ex:
            print("Error: File not found! Details: " + str(ex))

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

        if ("study" in listOfTokens[4]):
            activities = [1, 0, 0, 0, 0, 0, 0, 0]
        if ("read" in listOfTokens[4]):
            activities = [0, 1, 0, 0, 0, 0, 0, 0]
        if ("rest" in listOfTokens[4]):
            activities = [0, 0, 1, 0, 0, 0, 0, 0]
        if ("sleep" in listOfTokens[4]):
            activities = [0, 0, 0, 1, 0, 0, 0, 0]
        if ("laptop/TV" in listOfTokens[4]):
            activities = [0, 0, 0, 0, 1, 0, 0, 0]
        if ("sport" in listOfTokens[4]):
            activities = [0, 0, 0, 0, 0, 1, 0, 0]
        if ("house-activities" in listOfTokens[4]):
            activities = [0, 0, 0, 0, 0, 0, 1, 0]
        if ("friends-night-at-home" in listOfTokens[4]):
            activities = [0, 0, 0, 0, 0, 0, 0, 1]
        currentEntry += activities

        return np.array(currentEntry)

    def writeConfigToFile(self, email):

        jsonModel = self._model.to_json()

        if (email == ""):
            filePath = self._path + '/saved-models/model.json'
            weightsPath = self._path + '/saved-models/weights.h5'
        else:
            filePath = self._path + '/saved-models/model-' + email + '.json'
            weightsPath = self._path + '/saved-models/weights-' + email + '.h5'

        with open(filePath, 'w') as file_descriptor:
            try:
                file_descriptor.write(jsonModel)
            except IOError:
                print('Could not write to file!')
        self._model.save_weights(weightsPath)

    def readConfigFromFile(self, email):
        modelExists = False
        if (email != ""):
            filePath = self._path + '/saved-models/model-' + email + '.json'
            weightsPath = self._path + '/saved-models/weights-' + email + '.h5'
            modelExists = os.path.exists(filePath)

        if not modelExists:
            filePath = self._path + '/saved-models/model.json'
            weightsPath = self._path + '/saved-models/weights.h5'

        try:
            dumpedModelFile = open(filePath, 'r')
        except IOError:
            print("Could not open from file to read!")
            return False

        try:
            dumpedModel = dumpedModelFile.read()
        except IOError:
            print("Could not read from file!")
            return False

        dumpedModelFile.close()

        self._model = keras.models.model_from_json(dumpedModel)
        self._model.load_weights(weightsPath)
        return True


    def plotStatistics(self, historyTraining):
        histTraining = pd.DataFrame(historyTraining.history)
        histTraining['epoch'] = historyTraining.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error and Loss')
        plt.plot(histTraining['epoch'], histTraining['mean_absolute_error'], label="Training Error")
        plt.plot(histTraining['epoch'], histTraining['loss'], label="Training Loss")
        plt.legend()
        plt.ylim([0, 0.07])

        plt.show()