#!/usr/bin/env python3
"""
Library containing class `Dataset` for CIFAR-100 dataset preparation.
"""


import pickle
import tarfile

import numpy as np

# Paths to unziped dataset
CIFAR_TRAIN = "cifar-100-python/train"
CIFAR_TEST = "cifar-100-python/test"
CIFAR_META = "cifar-100-python/meta"

# Names of superclasses to learn:
# for classification to superclass
SUPERCLASSES = ["fish", "insects"]
# for classification to fine classes of one superclass
FINE_SUPERCLASS = ["reptiles"]

# Image size:
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3 


class Dataset:
    """
    Class to preprocess dataset.
    """

    def __init__(self, args, filename, superclassFine=False):
        """
        Unzips the dataset, chooses required classes and stores data and targets
        to variables:
            `self.train_data`, `self.train_target`,
            `self.test_data`, `self.test_target`
            - params:   `args` - program arguments
                        `filename` - name of the unziped file with dataset
                        `supeclassFine` - whether we want to train fine classes
                                        (default: train superclasses)
        """

        self.num_classes = None
        self.train_data = None
        self.train_target = None
        self.test_data = None
        self.test_target = None

        # Unzip the dataset and load the data into dictionaries (keys can be found
        # on the website of the dataset).
        self.untar(filename)
        trainData = self.unpickle(CIFAR_TRAIN)
        testData = self.unpickle(CIFAR_TEST)
        metaData = self.unpickle(CIFAR_META)
        
        # decide whether use fine class or superclass dataset
        if superclassFine:
            # fine class
            self.createDataset((trainData, testData, metaData),
                                            FINE_SUPERCLASS, fineClasses=True)
        else:
            # superclass
            self.createDataset((trainData, testData, metaData),
                                            SUPERCLASSES)

        # Reshape data to (num_data, channels, height, width) for data transformation.
        # Because np starts filling axes from right to left 
        # (in order: row -> column -> channel)
        self.train_data = np.reshape(self.train_data, 
                                    (self.train_data.shape[0], IMAGE_CHANNELS,
                                    IMAGE_HEIGHT, IMAGE_WIDTH))
        self.test_data = np.reshape(self.test_data,
                                    (self.test_data.shape[0], IMAGE_CHANNELS,
                                    IMAGE_HEIGHT, IMAGE_WIDTH))

        # Shift channel axis on the last coordinate (for grayscale transformation)
        self.train_data = np.transpose(self.train_data, axes=[0,2,3,1])
        self.test_data = np.transpose(self.test_data, axes=[0,2,3,1])

    def untar(self, filename):
        """
        Unzips the dataset file and stores unziped version into 
        current working directory.
            - params:   `filename` - name of the dataset file
        """
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()


    def unpickle(self, filename):
        """
        Reads the unziped dataset file (train, test or metadata) 
        and stores its data into directory.
            - params:   `filename` - name of the dataset file

        Returns directory of data from dataset.
        """
        with open(filename, 'rb') as fo:
            # Using `latin1` encoding because it is recommended
            myDict = pickle.load(fo, encoding='latin1')

        return myDict
    

    def extractSubset(self, data, metaSuperClasses, superclassesNames, fineClasses):
        """
        Takes only interesting data based on the choosen supeclasses 
        and returns them.
            - params:   `data` - dataset to extract data from 
                                (dictionary with keys: `data`, 
                                `coarse_labels`, `fine_labels`)
                        `metaSuperClasses` - list of names of the superclasses
                        `supeclassesNames` - list of names of supeclasses 
                                            which we want to predict
                        `fineClasses` - whether we want to extract fine classes `True`, 
                                        superclasses `False`

        Returns prepared datas and targets (based on the superclasses).
        """

        # Get indices of the interesting superclasses (target values). 
        superclassesIndices = [metaSuperClasses.index(x) for x in superclassesNames]

        # All datas from which we want to extract.
        datas = np.array(data['data'])
        super_targets = data['coarse_labels']

        # Indides in the dataset of interesting data (choosen superclasses).
        indices = np.array([i for i, x in enumerate(super_targets)
                            if x in superclassesIndices])
        
        # Extract only interesting data.
        datas = np.take(datas, indices, axis=0)

        # Extract only interesting targets (either fine or super classes).
        targets = None
        if fineClasses:
            # Get fine classes
            targets = np.take(data['fine_labels'], indices)
        else:
            # Get superclasses
            targets = np.take(super_targets, indices)

        self.num_classes = np.unique(targets).shape[0]

        return datas, targets


    def createDataset(self, data, superclassesNames, fineClasses=False):
        """
        Creates dataset from the raw data loaded from the dataset file.
        Extracts interesting data (from given superclasses) 
        and sets correct targets and stores it into proper `self` variables.
        """

        train = data[0]
        test = data[1]
        meta = data[2]['coarse_label_names']

        # Extract data and set the proper parameters.
        self.train_data, self.train_target = self.extractSubset(train, meta,
                                                                superclassesNames,
                                                                fineClasses)
        self.train_target = self.renameTargets(self.train_target)
    
        self.test_data, self.test_target = self.extractSubset(test, meta,
                                                                superclassesNames,
                                                                fineClasses)
        self.test_target = self.renameTargets(self.test_target)


    def renameTargets(self, targets):
        """
        Renames target names to interval [0..(num_classes - 1)] for better 
        data manipulation.
        The mapping:    lowest target label -> 0
                        highest target label -> num_classes - 1
                        other: by induction (+1)
        """
        unique_targets = np.unique(targets)
        for i, x in enumerate(unique_targets):     
            targets[targets==x] = i

        return targets