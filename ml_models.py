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



class Modeling:
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




    def SVM(self,c=1,weights='balanced',kernel='linear',probability=False):
        '''
        c :argument (optional) c value  for SVM
        weights :argument (optional) svm weights across classes
        kernel :argument (optional) svm kernel
        probability :argument (optional) if True estimates also probabilities. SLower but give probability access during testing

        NOTE: Trained model is stored in self.model


        Example: clf.SVM() or clf.SVM(c=50,weights=dict,kernel='rbf'),  where clf an instance of ml_models.Modeling()
        More details on argument values: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        :return: trained SVM model
        '''
        from sklearn.svm import SVC

        clf = SVC(C= c,class_weight= weights,kernel=kernel,probability=probability)
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


        Example: clf.DeciosionTree() or clf.DeciosionTree(criterion='gini',weights=dict), where clf an instance of ml_models.Modeling()

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

            Example: clf.RandomForest() or clf.DeciosionTree(criterion='gini',weights=dict), where clf an instance of ml_models.Modeling()

            More details on argument values:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

            :return: trained Random Forest model
            '''
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                         class_weight=weights, random_state=random_state, max_features=max_features,warm_start=warm_start,bootstrap=bootstrap)
            clf.fit(self.X_train, self.Y_train)

            self.name = 'randomforest'
            self.model = clf
            self.params = clf.get_params()
            self.params['feature_importances'] = clf.feature_importances_


    def GradientBoosting(self,loss='deviance',learning_rate=0.1, n_estimators=100, criterion='friedman_mse', max_depth=100, random_state=None, max_features='auto'):
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

        Example: clf.RandomForest() or clf.DeciosionTree(criterion='gini',weights=dict), where clf an instance of ml_models.Modeling()

        More details on argument values:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

        :return: trained Gradient Boosted Tree model
        '''
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier(loss=loss,learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state, max_features=max_features)
        clf.fit(self.X_train, self.Y_train)

        self.name = 'gradientboosting'
        self.model = clf
        self.params = clf.get_params()
        self.params['feature_importances'] = clf.feature_importances_


    def LRegression(self,fit_intercept=True,normalize=True,copy_X=False,n_jobs=None):
        '''
        fit_intercept :argument (optional) If 'True' feature value are being centered around zero
        normalize :argument (optional) If 'True' performs standard normalization ((x-m)/std)
        copy_X :argument (optional) If 'False' processing happens in place
        n_jobs :argument (optional) multiprocessing option

        NOTE: Trained model is stored in self.model


        Example: reg.LRegression(fit_intercept=False,normalize=False,n_jobs=10), where reg an instance of ml_models.Modeling()
        More details on argument values: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        :return: trained Linear Regression model
        '''
        from sklearn.linear_model import LinearRegression

        reg = LinearRegression(fit_intercept=fit_intercept,normalize=normalize,copy_X=copy_X,n_jobs=n_jobs)
        reg.fit(self.X_train, self.Y_train)

        self.name = 'linearregression'
        self.model = reg
        self.params = reg.get_params()



    def ElasticNetLRegression(self,fit_intercept=True,normalize=True,copy_X=False,max_iter=1000,alpha=1,l1_ratio=0.5,warm_start=False,positive=False):
        '''
        fit_intercept :argument (optional) If 'True' feature value are being centered around zero
        normalize :argument (optional) If 'True' performs standard normalization ((x-m)/std)
        copy_X :argument (optional) If 'False' processing happens in place
        max_iter :argument (optional) number of iterations
        alpha :argument (optional) constant to multiply or independent variables
        l1_ratio :argument (optional) l1_ratio=0 --> L1 Regularization, l1_ratio=1 --> L2 Regularization, all others are elastic net
        warm_start :argument (optional) 'True'--> previous iteration's results are used as initialization point, 'False'--> train from scratch
        positive :argument (optional) 'True'--> force foeeffs to be positive
        NOTE: Trained model is stored in self.model


        Example: reg.LRegression(fit_intercept=False,normalize=False,n_jobs=10), where reg an instance of ml_models.Modeling()
        More details on argument values: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        :return: trained Linear Regression model
        '''
        from sklearn.linear_model import ElasticNet

        reg = ElasticNet(fit_intercept=fit_intercept,normalize=normalize,copy_X=copy_X,max_iter=max_iter,alpha=alpha,l1_ratio=l1_ratio,warm_start=warm_start,positive=positive)
        reg.fit(self.X_train, self.Y_train)

        self.name = 'elasticnetregression'
        self.model = reg
        self.params = reg.get_params()



    def  HMMGmm(self,lengths,n_components,startprob=0,transmat=0,n_mix=5,covariance_type="tied",algorithm='map'):
        '''
        Train HMM with Gaussian Mixtures
        :param lengths: list with length of each sequence
        :param n_components: number of components = #classes
        :param startprob: (optional)list with starting probabilities for each component, pass
        :param transmat: (optional) transition probabilities matirix of size [#classes,#classes] - each row must sum to one
        :param n_mix: (optional) number of gaussians in a mixture
        :param covariance_type: (optional) type of covariance matrix : tied,full, or diag
        :param algorithm: (optional) inference algorithm to bve used 'map' or 'viterbi'

        NOTE: Trained model is stored in self.model


        Example:  model.HMMGmm (lengths=counts_ordered,startprob=start_probs,transmat=trans_probs,n_components=start_probs.shape[0])
        More details on argument values: https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.hmm.GMMHMM

        :return: trained Linear Regression model
        '''
        init_params = ""
        from hmmlearn import hmm
        if startprob==0:
            init_params+='s'
        if transmat==0:
                init_params+='t'

        hmm_model = hmm.GMMHMM ( n_components=n_components,n_mix=n_mix, covariance_type=covariance_type,init_params=init_params,startprob_prior=startprob,transmat_prior=transmat,algorithm=algorithm )
        hmm_model.fit(self.X_train, lengths)
        self.name = 'hmm_gmm'
        self.model = hmm_model
        self.params = {'transition_matrix':transmat,'startprobs':startprob}
        
        

    def SaveModel(self,output):
        import pickle5 as pickle
        with open(output, 'wb') as handle:
            classifier = {}
            classifier['model'] = self.model
            classifier['parameters'] = self.params
            classifier['results'] = self.results
            classifier['results'] = self.name
            pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Trained '+ self.name.upper()+' model Saved in:',output)



















