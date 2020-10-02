# sklearn_wrapper
A wrapper to simplify the usage of sklearn for massive experimentation with different modeling methods that provides easy access to the model parameters

Currently one class is supported for classification methods

## Modeling Class
####  Methods

* SVM
* Random Forest
* Decision Tree
* Gradient Boosting 
* Linear Regression
* Elastic Net Regression
* HMM with GMM Probabilities - requires [hmmlearn lib](https://pypi.org/project/hmmlearn/)
#### Other Methods 

* Save Model
  * Stores model name
  * Model Hyperparameters
  * The actual Model
  * The performance results if the model has been tested and results have been added into model.results dict (example:  model.results['accuracy'] = accuracy)


## More methods and functionalities will be added gradually in the near future

