# sklearn_wrapper
A wrapper to simplify the usage of sklearn for massive experimentation with different modeling methods that provides easy access to the model parameters

Currently one class is supported for classification methods

## Modeling Class
#### Init Method 
* You should pass the train data and labels when initializing. The you can run all modeling methods on these data
* Optionally you can also pass the test data and labels so they are alighed and always be tested together

####  MOdeling Methods

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
  * The performance results if the model has been tested and results have been added into model.results dict - examples:
   ```python
   model.results['accuracy'] = accuracy_value
   model.results['r^2'] = rSquared_value
   model.results['custom_metric_name'] = custom_metric_value
   ```
  * #### Will update soon to optionally store the train and test data and labels
  
## More methods and functionalities will be added gradually in the near future

