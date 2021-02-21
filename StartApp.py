from ANN.NormalizedDataset import NormalizedDataset
from ANN.ANN import ANN
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    ann = ANN("")

    print("Menu:\n")
    print("1. Train the model with the initial dataset\n")
    print("2. Validate with 1st test data sample \n")
    print("3. Validate with 2nd test data sample \n")

    command = input("Choose command: \n")
    if int(command) == 1:
       dataset = NormalizedDataset(
        "datasets/initialDataset.txt")
        historyTraining = ann.train(dataset.getSampleMatrix(), dataset.getVerdictsVector())
        ann.plotStatistics(historyTraining)
        saveModelToPickle(ann)
    elif int(command) == 2:
        datasetValidation = NormalizedDataset(
        "/test-datasets/testDataset-initialPhase.txt")
        ann.validate(datasetValidation.getSampleMatrix(), datasetValidation.getVerdictsVector())
    elif int(command) == 3:
        datasetValidationFinalPhase = NormalizedDataset(
        "test-datasets/testDataset-finalPhase.txt")
        ann.validate(datasetValidationFinalPhase.getSampleMatrix(), datasetValidationFinalPhase.getVerdictsVector())
    else:
        print("Invalid command! \n")


def saveModelToPickle(ann):
    flag = ''
    while flag != 'y' and flag != 'n':
        flag = input("Write current model to file? [y/n]")
        if flag == 'y':
            ann.writeConfigToFile("")
        elif flag != 'n':
            print('Invalid input! Try again...')
        else:
            break


if __name__ == "__main__":
    main()
