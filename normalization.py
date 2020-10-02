'''
COPYRIGHTS
Michalis Papakostas
Postdoctoral Fellow
September 2020
University of Michigan - Ann Arbor
'''


class Preprocessing:
    '''
    X_train :argument - training data as a numpy array
    Y_train :argument - trainig labels as a numpy array
    X_test :argument (optional) - test data as numpy array
    Y_test :argument (optional) - test labels as numpy array

    NOTE 1: self.params stores all the parameters set for training the classifier
    NOTE 2: self.results stores all the results for testing the classifier with parameters = self.params
    NOTE 3: self.model stores the trained model
    NOTE 4: self.name stores the name of the classifier
    Example:  clf = classifiers.Classification(X_train,Y_train) or  clf = classifiers.Classification(X_train,Y_train,X_test,Y_test)
    '''

    def __init__(self, X_train, Y_train, **kargs):
        self.X_train = X_train
        self.Y_train = Y_train
        try:
            self.X_test = kargs['X_test']
            self.Y_test = kargs['Y_test']
        except:
            pass
        try:
            del kargs['X_test']
            del kargs['Y_test']
        except:
            pass
        self.params = {}
        self.results = {}
        self.model = {}
        self.name = None
