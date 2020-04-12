# Script output:
In descending order
```
MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True): Score: 0.756
MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True): Score: 0.755
MultinomialNB(alpha=1, class_prior=None, fit_prior=True): Score: 0.748
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(10,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False): Score: 0.739
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False): Score: 0.737
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False): Score: 0.597
 ```
 ##Note:
 This script takes about 3 minutes to run mainly due to generating the tokenized dataset and the Nearal Net with 
 hidden layers of size 5.