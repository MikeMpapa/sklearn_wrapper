'''
COPYRIGHTS
Michalis Papakostas
Postdoctoral Fellow
March 2020
University of Michigan - Ann Arbor
'''

import numpy

#from scipy.spatial import distance
#from hmmlearn import hmm
#import keras
#from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Conv3D
#from keras.models import Sequential



class Classification:
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




    def SVM(self,c=1,weights='balanced',kernel='linear'):
        '''
        c :argument (optional) c value  for SVM
        weights :argument (optional) svm weights across classes

        NOTE: Trained model is stored in self.model


        Example: clf.SVM() or clf.SVM(c=50,weights=dict,kernel='rbf'), were clf an instance of classifiers.Classification()
        More details on argument values: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        :return: trained SVM model
        '''
        from sklearn.svm import SVC

        clf = SVC(C= c,class_weight= weights,kernel=kernel)
        clf.fit(self.X_train, self.Y_train)

        self.name = 'svm'
        self.model = clf
        self.params = clf.get_params()

    def DeciosionTree(self,criterion='entropy',splitter='best',max_depth=100,random_state=None,weights='balanced',max_features='auto'):
        '''
        criterion :argument (optional) critirion to split at each node
        weights :argument (optional) decision tree weights across classes
        splitter :argument (optional) strategy used to split at each node
        max_depth :argument (optional) maximum tree depth, the larger the slower the tr
        random_state :argument (optional) initial classifier seed
        max_features :argument (optional) maximum number of features to consiter for an internal split

        NOTE 1: Trained model is stored in self.model
        NOTE 2: self.parameters includes also the feature importances


        Example: clf.DeciosionTree() or clf.DeciosionTree(criterion='gini',weights=dict), were clf an instance of classifiers.Classification()

        More details on argument values: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

        :return: trained Decision Tree model
        '''
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,class_weight=weights,random_state=random_state,max_features=max_features)
        clf.fit(self.X_train, self.Y_train)

        self.name = 'decisiontree'
        self.model = clf
        self.params = clf.get_params()
        self.params['feature_importances'] = clf.feature_importances_


    def RandomForest(self,n_estimators=50,criterion='entropy', max_depth=100, random_state=None,
                         weights=None, max_features='auto',warm_start=True,bootstrap=True):
            '''
            criterion :argument (optional) critirion to split at each node
            weights :argument (optional) decision tree weights across classes
            max_depth :argument (optional) maximum tree depth, the larger the slower the tr
            random_state :argument (optional) initial classifier seed
            max_features :argument (optional) maximum number of features to consiter for an internal split
            warm_start :argument (optional) whether to use previous esstimator as a starting  point or build new estimator from scratch
            bootstrap :argument (optional) whether to use the whole dataset or part of it to build each estimator

            NOTE 1: Trained model is stored in self.model
            NOTE 2: self.parameters includes also the feature importances

            Example: clf.RandomForest() or clf.DeciosionTree(criterion='gini',weights=dict), were clf an instance of classifiers.Classification()

            More details on argument values:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

            :return: trained Decision Tree model
            '''
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                         class_weight=weights, random_state=random_state, max_features=max_features,warm_start=warm_start,bootstrap=bootstrap)
            clf.fit(self.X_train, self.Y_train)

            self.name = 'randomforest'
            self.model = clf
            self.params = clf.get_params()
            self.params['feature_importances'] = clf.feature_importances_



    def SaveModel(self,output):
        import pickle
        with open(output, 'wb') as handle:
            classifier = {}
            classifier['model'] = self.model
            classifier['parameters'] = self.params
            pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)


















