# porto-seguro-imbalanced-comparison-classifiers

**Overview:** 

(according to this Porto Seguro's Safe Driver Prediction Dataset: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)

“Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.”

**Data Description:**

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder."

**Scope:**

Use different classification methods to predict more accurately how many auto insurance policy holder files a claim (predict the probability) while avoid overfitting.

**How to proceed:**

Use different models such as Single Decision Tree, Ensemble Classifiers (Bagging, Balanced Bagging, Random Forest, Balanced Random Forest, Easy Ensemble Classifier, RUS Boost), XGBoost, Deep Neural Network to evaluate their performances on both imbalanced Train and Test set while avoid fitting.

Different metrics such as Accuracy (we do not rely on this one), Balanced Accuracy, Geometric Mean, Precision, Recall, F1 score, Confusion Matrix will be calculated.

Find the most important features of this dataset which can be used in policies to predict number of ‘Not Claim’ or ‘Claim’ customers after applying those new changes.

**Problem statement:** 

Imbalanced dataset could cause overfitting. We can not rely on Accuracy as a metric for imbalanced dataset (will be usually high and misleading) so we would use confusion matrix, balanced accuracy, geometric mean, F1 score instead. 

**Target statement:**

Selecting the best classification method which also can avoid overfitting.

**Target achievement:**

RUS Boost has the highest Balanced Accuracy, Geometric Mean, F1 score and the best Confusion Matrix among all classification methods.


imbalanced-learn offers a number of re-sampling techniques commonly used in strong between-class imbalanced datasets. This Python package is also compatible with scikit-learn.

In my case, I got the same error "dot: graph is too large for cairo-renderer bitmaps. scaling by 0.0347552 to fit" all of the time when running the Balanced Random Forest on old version of Colab Notebooks. Here are dependencies of imbalanced-learn:

Python 3.6+

scipy(>=0.19.1)

numpy(>=1.13.3)

scikit-learn(>=0.23)

joblib(>=0.11)

keras 2 (optional)

tensorflow (optional)

matplotlib(>=2.0.0)

pandas(>=0.22)

**Installation:** 

You should install imbalanced-learn on the PyPi's repository via pip from the begging and restart the runtime, then start your work:
```pip install -U imbalanced-learn```

Anaconda Cloud platform: 

```conda install -c conda-forge imbalanced-learn```

Here are Classification methods which I would create and evaluate in my file:

**Single Decision Tree** 

**Ensemble classifier using samplers internally:**

- Easy Ensemble classifier [1]

- Random Forest 

- Balanced Random Forest [2] 

- Bagging (Classifier)

- Balanced Bagging [3]

- Easy Ensemble [4]

- RUSBoost [5]

**Mini-batch resampling for Keras and Tensorflow (Deep Neural Network - MLP) [6]**


**Table of Contents:**

**Comparison of ensembling classifiers internally using sampling**

**A. Data Engineering:**

A.1. Load libraries

A.2. Load an imbalanced dataset

A.3. Data Exploration

A.4. Check Missing or Nan

A.5. Create X, y

A.6. One hot encoding [7] (One hot encoding is not ideally fit for Ensemble Classifiers so next time I will try to use Label Encoding for these kinds of imbalanced dataset instead.)

A.7. Split data

A.8. Unique values of each features

A.9. Draw Pairplot

A.10. Confusion Matrix Function

**B. Comparison of Ensemble Classifiers [8], XGBoost Classifier [9][10][11], Deep Neural Network (Mini-batch resampling for Keras and Tensorflow)**

- Confusion Matrix

- Mean ROC AUC

- Accuracy scores on Train / Test set (We should not rely on accuracy as it would be high and misleading. Instead, we should look at other metrics as confusion matrix, Balanced accuracy, Geometric mean, Precision, Recall, F1-score.

- Classification report (Accuracy, Balanced accuracy, Geometric mean, Precision, Recall, F1-score)

**C. Feature Importance**

**D. Heatmap**

**E. Draw Single Decision Tree**

**F. ROC & AUC between Deep Neural Network, Ensemble Classifiers, XGBoost Classifier**

**G. Predict**

**H. New Policy on Trial:**

H.1 List out

H.2 Implement that New Policy

H.3 Result


**References:**
[1] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

[2] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html

[3] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedBaggingClassifier.html

[4] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

[5] https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.RUSBoostClassifier.html

[6] https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/applications/porto_seguro_keras_under_sampling.html?highlight=mini%20batch

[7] https://www.reddit.com/r/MachineLearning/comments/ihwbsn/d_why_onehot_encoding_is_a_poor_fit_for_random/?utm_source=share&utm_medium=ios_app&utm_name=iossmf

[8] Comparison of ensembling classifiers internally using sampling. https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/ensemble/plot_comparison_ensemble_classifier.html#sphx-glr-auto-examples-ensemble-plot-comparison-ensemble-classifier-py

[9] https://machinelearningmastery.com/xgboost-for-imbalanced-classification/

[10] https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost

[11] https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
