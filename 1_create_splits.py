from pathlib import Path
import sys 
from dataUtil import DataUtil 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def createRandomSplits(datasets,modality,outputfilename):
    subpaths = [] 
    for ds in datasets:
        inputfolder = fr"..\Data\{ds}\{modality}\1_Original_Organized"
        subpaths.extend( DataUtil.getSubDirectories(inputfolder))

    cases = [x.stem.replace("_L1","") for x in subpaths if (("L2" not in str(x)) and ("L3" not in str(x)) and ("L4" not in str(x)))  ]

    X_train, X_test = train_test_split(cases,test_size=0.5)

    X_train, X_val = train_test_split(X_train,test_size=0.5)

    splitsdict = {}

    for item in X_train:
        splitsdict[item] = "train"

    for item in X_test:
        splitsdict[item] = "test"

    for item in X_val:
        splitsdict[item] = "val"

    outputfolder = fr"outputs\splits"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    DataUtil.writeJson(splitsdict,fr"{outputfolder}\{outputfilename}.json")


def createTrainValSplits(datasets,modality,outputfilename):
    subpaths = [] 
    for ds in datasets:
        inputfolder = fr"..\Data\{ds}\{modality}\1_Original_Organized"
        subpaths.extend( DataUtil.getSubDirectories(inputfolder))

    cases = [x.stem.replace("_L1","") for x in subpaths if (("L2" not in str(x)) and ("L3" not in str(x)) and ("L4" not in str(x)))  ]

    X_train, X_test = train_test_split(cases,test_size=0.5)

    splitsdict = {}

    for item in X_train:
        splitsdict[item] = "train"

    for item in X_test:
        splitsdict[item] = "test"

    outputfolder = fr"outputs\splits"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    DataUtil.writeJson(splitsdict,fr"{outputfolder}\{outputfilename}.json")

def createTestSplit(datadir, datasets,modality,outputfilename):
    subpaths = [] 
    for ds in datasets:
        inputfolder = fr"..\Data\{ds}\{modality}\1_Original_Organized"
        subpaths.extend( DataUtil.getSubDirectories(inputfolder))

    cases = [x.stem.replace("_L1","") for x in subpaths if (("L2" not in str(x)) and ("L3" not in str(x)) and ("L4" not in str(x)))  ]

    splitsdict = {}

    for item in cases:
        splitsdict[item] = "test"

    outputfolder = fr"outputs\splits"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    DataUtil.writeJson(splitsdict,fr"{outputfolder}\{outputfilename}.json")


if __name__ == "__main__":
    
    datadir = fr"..\Data"
    
    modality = fr"T2W"
    datasets = ["...."]

    outputfilename = fr"....."

    createTestSplit(datadir,datasets,modality,outputfilename)

