# Porto Seguro's Safe Driver Prediction - Comparison of multiple Classfifers on the imbalanced dataset

## Category:

- Banking

- Financial Institute

- Insurance

## Language:

- Python

**Overview:** 

(according to this Porto Seguro's Safe Driver Prediction dataset: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)

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

<img src="https://user-images.githubusercontent.com/70437668/141063635-31543d54-8d68-46ee-b064-051b6f639583.jpg" width=50% height=50%>

A.3. Data Exploration

A.4. Check Missing or Nan

A.5. Create X, y

A.6. One hot encoding [7] (One hot encoding is not ideally fit for Ensemble Classifiers so next time I will try to use Label Encoding for these kinds of imbalanced dataset instead.)

A.7. Split data

A.8. Unique values of each features

A.9. Draw Pairplot

```
sns.set()
sns.pairplot(df_train, hue='ps_ind_03')
```

A.10. Confusion Matrix Function

```
def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
```

**B. Comparison of Ensemble Classifiers [8], XGBoost Classifier [9][10][11], Deep Neural Network (Mini-batch resampling for Keras and Tensorflow)**

- Confusion Matrix

- Mean ROC AUC

- Accuracy scores on Train / Test set (We should not rely on accuracy as it would be high and misleading. Instead, we should look at other metrics as confusion matrix, Balanced accuracy, Geometric mean, Precision, Recall, F1-score.

- Classification report (Accuracy, Balanced accuracy, Geometric mean, Precision, Recall, F1-score)

## Single Decision Tree 

We use the training of Single Decision Tree classifier as a baseline to compare with other classifiers on this imbalanced dataset.

Balanced accuracy and geometric mean are reported followingly as they are metrics widely used in the literature to validate model trained on imbalanced set.

<img src="https://user-images.githubusercontent.com/70437668/141063584-8b33093a-95d4-490a-8c99-622e3b318897.jpg" width=50% height=50%>

```
Mean ROC AUC on Train Set: 0.506
Mean ROC AUC on Test Set: 0.508
```

```
Single Decision Tree score on Train Set: 1.0
Single Decision Tree score on Test Set: 0.918516153962467
```

## Bagging & Balanced Bagging

A number of estimators are built on various randomly selected data subsets in ensemble classifiers. But each data subset is not allowed to be balanced by Bagging classifier because the majority classes will be favored by it when implementing training on imbalanced data set.

In contrast, each data subset is allowed to be resample in ordor to have each ensemble's estimator trained by the Balanced Bagging Classifier. This means the output of an Easy Ensemble sample with an ensemble of classifiers, Bagging Classifier for instance will be combined. So an advantage of Balanced Bagging Classifier over Bagging Classifier from scikit learn is that it takes the same parameters and also another two parameters, sampling stratgy and replacement to keep the random under-sampler's behavior under control.

<img src="https://user-images.githubusercontent.com/70437668/141063584-8b33093a-95d4-490a-8c99-622e3b318897.jpg" width=50% height=50%>

### Bagging
```
Mean ROC AUC on Train Set: 0.537
Mean ROC AUC on Test Set: 0.539
```

### Balanced Bagging
```
Mean ROC AUC on Train Set: 0.572
Mean ROC AUC on Test Set: 0.559
```

```
Bagging Classifier score on Train Set: 0.991356010156058
Balanced Bagging Classifier score on Test Set: 0.8057660321567178
```

## Random Forest & Balanced Random Forest 

Random Forest is another popular ensemble method and it is usually outperforming bagging. Here, we used a vanilla random forest and its balanced counterpart in which each bootstrap sample is balanced.

<img src="https://user-images.githubusercontent.com/70437668/141063505-ab1d7cbb-dd20-4220-b51a-5edb2a0369f9.jpg" width=50% height=50%>

### Random Forest
```
Mean ROC AUC on Train Set: 0.523
Mean ROC AUC on Test Set: 0.521
```

### Balanced Random Forest
```
Mean ROC AUC on Train Set: 0.551
Mean ROC AUC on Test Set: 0.539
```

```
Random Forest classifier score on Train Set: 0.9914799157442
Balanced Random Forest classifier score on Test Set: 0.5403807059693218
```

## Easy Ensemble & RUS Boost

In the same manner, Easy Ensemble classifier is a bag of balanced AdaBoost classifier. However, it will be slower to train than random forest and will achieve worse performance

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

RUS Boost: Several methods taking advantage of boosting have been designed. RUSBoostClassifier randomly under-sample the dataset before to perform a boosting iteration. Random under-sampling integrating in the learning of an AdaBoost classifier. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm.

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.RUSBoostClassifier.html?highlight=rusboost#imblearn.ensemble.RUSBoostClassifier

<img src="https://user-images.githubusercontent.com/70437668/141063470-03a867db-4329-4f36-ae7c-77fec93ada2a.jpg" width=50% height=50%>

### Easy Ensemble
```
Mean ROC AUC on Train Set: 0.608
Mean ROC AUC on Test Set: 0.603
```

```
Easy Ensemble Classifier score on Train Set: 0.6258828273155119

```

### RUS Boost
```
Mean ROC AUC on Train Set: 0.627
Mean ROC AUC on Test Set: 0.613
```

```
RUS Boost score on Test Set: 0.6154802506678315
```

## XGBoost

XGBoost provides a highly efficient implementation of the stochastic gradient boosting algorithm and access to a suite of model hyperparameters designed to provide control over the model training process.

https://machinelearningmastery.com/xgboost-for-imbalanced-classification/

<img src="https://user-images.githubusercontent.com/70437668/141063439-e46b1a8a-b218-418a-94de-46fd635e4597.jpg" width=50% height=50%>

```
Mean ROC AUC on Train Set: 0.620
Mean ROC AUC on Test Set: 0.589
```

```
XGBoost classifier score on Test Set: 0.9633238688866115
```

## Deep Neural Network's result

```
Epoch 5/5
477/477 [==============================] - 4s 8ms/step - loss: 0.1579 - accuracy: 0.9634 - val_loss: 0.1514 - val_accuracy: 0.9642
```

y_pred
```
[[0.02850922]
 [0.00979213]
 [0.05552964]
 ...
 [0.00529212]
 [0.00812945]
 [0.00510255]]
```

![DNN result](https://user-images.githubusercontent.com/70437668/141063412-8d83784c-8f47-4dcd-94e6-05498598ad43.jpg)

**C. Feature Importance**

## Decision Tree

![Decision Tree Feature Importance](https://user-images.githubusercontent.com/70437668/141063370-2436a59f-680b-455d-bb2f-9913021f6a70.jpg)

## Random Forest 

![Random Forest Feature Importance](https://user-images.githubusercontent.com/70437668/141063361-a29a1527-fb9d-4d36-8642-91d36c2f18a7.jpg)

## Balanced Random Forest

![Balanced Random Forest Feature Importance](https://user-images.githubusercontent.com/70437668/141063356-b7168576-5acd-4242-bc91-49d9e8f4b46c.jpg)

## RUS Boost

![RUS Boost Feature Importance](https://user-images.githubusercontent.com/70437668/141063346-b28aec1b-9463-47f8-88b5-1d0a61622b7f.jpg)

## XGBoost

![XGBoost Feature Importance](https://user-images.githubusercontent.com/70437668/141063338-e60ea688-48f9-4df0-8d13-5110c664b978.jpg)


**D. Heatmap**

## Train set

<img src="https://user-images.githubusercontent.com/70437668/141063258-0d1b0dec-d7e5-4bc3-a5f2-8f232aff86da.jpg" width=50% height=50%>

## Test set

<img src="https://user-images.githubusercontent.com/70437668/141063268-814e866d-7ad9-4fe5-acc3-89f2635d0ff9.jpg" width=50% height=50%>

**E. Draw Single Decision Tree**

![download](https://user-images.githubusercontent.com/70437668/141065326-d2dd2570-b789-4a08-8aa5-7ccf4f2cbdd4.png)

**F. ROC & AUC between Deep Neural Network, Ensemble Classifiers, XGBoost Classifier**

RUS Boost has the highest ROC AUC = 0.624 to be considered the best Classifier on the imbalanced dataset.

```
No Skill: ROC AUC=0.500
With MLP: ROC AUC=0.621
With Decision Tree: ROC AUC=0.503
With Bagging: ROC AUC=0.535
With Balanced Bagging: ROC AUC=0.560
With Random Forest: ROC AUC=0.521
With Balanced Random Forest: ROC AUC=0.549
With Easy Ensemble: ROC AUC=0.612
With RUS Boost: ROC AUC=0.624
With XGBoost: ROC AUC=0.609
Best: ROC AUC=1.000
```

<img src="https://user-images.githubusercontent.com/70437668/141063180-291c4d4c-69d6-41fc-9f9c-1d724e89ab66.jpg" width=50% height=50%>

**G. Predict**

![Predict](https://user-images.githubusercontent.com/70437668/141063151-3e227c9e-0a2b-4c0c-ba4f-97442010813a.jpg)

**H. New Policy on Trial:**

H.1 List out

H.2 Implement that New Policy
```
result = dectree.predict(new_policy)
len(np.where(result==1)[0])
```

Output
```
30537
```
H.3 Result

**30537 drivers will claim insurance instead of 3 with entropy = 0 when changing ps_car_13 (most influential feature by Single Decision Tree with max_depth=5) to 2.5 for example (as long as greater than 2.447).**

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
