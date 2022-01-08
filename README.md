# Porto Seguro's Safe Driver Prediction - Comparison of multiple Classfifers on the imbalanced dataset

## Category:

- Banking

- Financial Institute

- Insurance

## Language:

- Python

### Overview:

(according to this Porto Seguro's Safe Driver Prediction dataset: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)

“Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.”

In this project I applied what I have used in my previous project [Banking Dataset - Marketing Targets](https://github.com/daiphuongngo/Banking-Dataset-Imbalanced-Learn-Comparison) with all classiers as mentioned below and added 2 Supervised Machine Learning methods: Distributed Random Forest (DRF) & Gradient Boosting Machine (GBM) to overlook and compare performance between Tensorflow, Keras & H20 models.

### Data Description:

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder."

### Scope:

Use different classification methods to predict more accurately how many auto insurance policy holder files a claim (predict the probability) while avoid overfitting.

### How to proceed:

Use different models such as Single Decision Tree, Ensemble Classifiers (Bagging, Balanced Bagging, Random Forest, Balanced Random Forest, Easy Ensemble Classifier, RUS Boost), XGBoost, Deep Neural Network, Distributed Random Forest H2O, Gradient Boosting H2O to evaluate their performances on both imbalanced Train and Test set while avoid fitting.

Different metrics such as Accuracy (I do not rely on this one), Balanced Accuracy, Geometric Mean, Precision, Recall, F1 score, Confusion Matrix will be calculated.

Find the most important features of this dataset which can be used in policies to predict number of ‘Not Claim’ or ‘Claim’ customers after applying those new changes.

### Problem statement:

Imbalanced dataset could cause overfitting. I can not rely on Accuracy as a metric for imbalanced dataset (will be usually high and misleading) so I would use confusion matrix, balanced accuracy, geometric mean, F1 score instead. 

### Target statement:

Selecting the best classification method which also can avoid overfitting.

### Target achievement:

- RUS Boost had the highest Balanced Accuracy=0.59, Geometric Mean=0.58, best Confusion Matrix [[35428 21913][974 1206]], ROC AUC=0.624 among classifiers while the second-best classifier XGBoost had Balanced Accuracy=0.50, Geometric Mean=0.03, Confusion Matrix [[57336 5][2178 2]], ROC AUC=0.609.

- Dropped insurance claimers to 30,537 drivers when changing ps_car_13 (most influential feature by RUS Boost & Single Decision Tree with max_depth=5) to 2.5 as per RUS Boost, & further when ps_car_13 >2.447, entropy<0.57 as per Decision Tree.

-  Increased insurance claimers to 511,884 when changing ps_car_13 to 0.5 & 556,283 when ps_car_13=1 but fell to 491,936 when ps_car_13=-1.

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

h20

### Installation:

You should install imbalanced-learn on the PyPi's repository via pip from the begging and restart the runtime, then start your work:
```pip install -U imbalanced-learn```

Anaconda Cloud platform: 

```conda install -c conda-forge imbalanced-learn```

Here are Classification methods which I would create and evaluate in my file:

#### Single Decision Tree

#### Ensemble classifier using samplers internally:

- Easy Ensemble classifier [1]

- Random Forest 

- Balanced Random Forest [2] 

- Bagging (Classifier)

- Balanced Bagging [3]

- Easy Ensemble [4]

- RUSBoost [5]

- Mini-batch resampling for Keras and Tensorflow (Deep Neural Network - MLP) [6]

- Distributed Random Forest H2O [12]

- Gradient Boosting H20 [13]

--> I gave an attempt to use H20.ai with these 2 classifers for this project. They generated more detailed report in a much more rapid performance. This can be applied to all possible classifers in the latter stage.

### Table of Contents:

**Comparison of ensembling classifiers internally using sampling**

### A. Data Engineering:

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

### B. Comparison of Ensemble Classifiers [8], XGBoost Classifier [9][10][11], Deep Neural Network (Mini-batch resampling for Keras and Tensorflow)

- Confusion Matrix

- Mean ROC AUC

- Accuracy scores on Train / Test set (I should not rely on accuracy as it would be high and misleading. Instead, I should look at other metrics as confusion matrix, Balanced accuracy, Geometric mean, Precision, Recall, F1-score.

- Classification report (Accuracy, Balanced accuracy, Geometric mean, Precision, Recall, F1-score)

#### Single Decision Tree 

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

I use the training of Single Decision Tree classifier as a baseline to compare with other classifiers on this imbalanced dataset.

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

#### Bagging & Balanced Bagging

A number of estimators are built on various randomly selected data subsets in ensemble classifiers. But each data subset is not allowed to be balanced by Bagging classifier because the majority classes will be favored by it when implementing training on imbalanced data set.

In contrast, each data subset is allowed to be resample in ordor to have each ensemble's estimator trained by the Balanced Bagging Classifier. This means the output of an Easy Ensemble sample with an ensemble of classifiers, Bagging Classifier for instance will be combined. So an advantage of Balanced Bagging Classifier over Bagging Classifier from scikit learn is that it takes the same parameters and also another two parameters, sampling stratgy and replacement to keep the random under-sampler's behavior under control.

<img src="https://user-images.githubusercontent.com/70437668/141063584-8b33093a-95d4-490a-8c99-622e3b318897.jpg" width=50% height=50%>

##### Bagging

A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

```
Mean ROC AUC on Train Set: 0.537
Mean ROC AUC on Test Set: 0.539
```

##### Balanced Bagging

A Bagging classifier with additional balancing. This implementation of Bagging is similar to the scikit-learn implementation. It includes an additional step to balance the training set at fit time using a given sampler. This classifier can serves as a basis to implement various methods such as Exactly Balanced Bagging, Roughly Balanced Bagging, Over-Bagging, or SMOTE-Bagging.

```
Mean ROC AUC on Train Set: 0.572
Mean ROC AUC on Test Set: 0.559
```

```
Bagging Classifier score on Train Set: 0.991356010156058
Balanced Bagging Classifier score on Test Set: 0.8057660321567178
```

#### Random Forest & Balanced Random Forest 

Random Forest is another popular ensemble method and it is usually outperforming bagging. Here, I used a vanilla random forest and its balanced counterpart in which each bootstrap sample is balanced.

<img src="https://user-images.githubusercontent.com/70437668/141063505-ab1d7cbb-dd20-4220-b51a-5edb2a0369f9.jpg" width=50% height=50%>

#### Random Forest

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

```
Mean ROC AUC on Train Set: 0.523
Mean ROC AUC on Test Set: 0.521
```

#### Balanced Random Forest

A balanced random forest randomly under-samples each boostrap sample to balance it.

```
Mean ROC AUC on Train Set: 0.551
Mean ROC AUC on Test Set: 0.539
```

```
Random Forest classifier score on Train Set: 0.9914799157442
Balanced Random Forest classifier score on Test Set: 0.5403807059693218
```

#### Easy Ensemble & RUS Boost

In the same manner, Easy Ensemble classifier is a bag of balanced AdaBoost classifier. However, it will be slower to train than random forest and will achieve worse performance

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html

RUS Boost: Several methods taking advantage of boosting have been designed. RUSBoostClassifier randomly under-sample the dataset before to perform a boosting iteration. Random under-sampling integrating in the learning of an AdaBoost classifier. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm.

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.RUSBoostClassifier.html?highlight=rusboost#imblearn.ensemble.RUSBoostClassifier

<img src="https://user-images.githubusercontent.com/70437668/141063470-03a867db-4329-4f36-ae7c-77fec93ada2a.jpg" width=50% height=50%>

##### Easy Ensemble

Bag of balanced boosted learners also known as EasyEnsemble. This algorithm is known as EasyEnsemble [1]. The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.

```
Mean ROC AUC on Train Set: 0.608
Mean ROC AUC on Test Set: 0.603
```

```
Easy Ensemble Classifier score on Train Set: 0.6258828273155119

```

##### RUS Boost

Random under-sampling integrated in the learning of AdaBoost. During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm.
```
Mean ROC AUC on Train Set: 0.627
Mean ROC AUC on Test Set: 0.613
```

```
RUS Boost score on Test Set: 0.6154802506678315
```

#### XGBoost

"Boosting is a strong alternative to bagging. Instead of aggregating predictions, boosters turn weak learners into strong learners by focusing on where the individual models (usually Decision Trees) went wrong. In Gradient Boosting, individual models train upon the residuals, the difference between the prediction and the actual results. Instead of aggregating trees, gradient boosted trees learns from errors during each boosting round.

XGBoost is short for “eXtreme Gradient Boosting.” The “eXtreme” refers to speed enhancements such as parallel computing and cache awareness that makes XGBoost approximately 10 times faster than traditional Gradient Boosting. In addition, XGBoost includes a unique split-finding algorithm to optimize trees, along with built-in regularization that reduces overfitting. Generally speaking, XGBoost is a faster, more accurate version of Gradient Boosting.

Boosting performs better than bagging on average, and Gradient Boosting is arguably the best boosting ensemble. Since XGBoost is an advanced version of Gradient Boosting, and its results are unparalleled, it’s arguably the best machine learning ensemble that we have."

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

#### Deep Neural Network's result

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

#### Distributed Random Forest H2O

Reference: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html

Distributed Random Forest (DRF) is a powerful classification and regression tool. When given a set of data, DRF generates a forest of classification or regression trees, rather than a single classification or regression tree. Each of these trees is a weak learner built on a subset of rows and columns. More trees will reduce the variance. Both classification and regression take the average prediction over all of their trees to make a final prediction, whether predicting for a class or numeric value. (Note: For a categorical response column, DRF maps factors (e.g. ‘dog’, ‘cat’, ‘mouse) in lexicographic order to a name lookup array with integer indices (e.g. ‘cat -> 0, ‘dog’ -> 1, ‘mouse’ -> 2.)

The current version of DRF is fundamentally the same as in previous versions of H2O (same algorithmic steps, same histogramming techniques), with the exception of the following changes:

Improved ability to train on categorical variables (using the nbins_cats parameter)

Minor changes in histogramming logic for some corner cases

By default, DRF builds half as many trees for binomial problems, similar to GBM: it uses a single tree to estimate class 0 (probability “p0”), and then computes the probability of class 0 as 1.0−p0. For multiclass problems, a tree is used to estimate the probability of each class separately.

There was some code cleanup and refactoring to support the following features:

+ Per-row observation weights

+ N-fold cross-validation

+ DRF no longer has a special-cased histogram for classification (class DBinomHistogram has been superseded by DRealHistogram) since it was not applicable to cases with observation weights or for cross-validation.

##### Summary

```
Model Details
=============
H2ORandomForestEstimator :  Distributed Random Forest
Model Key:  DRF_model_python_1598981279148_1


Model Summary: 
number_of_trees	number_of_internal_trees	model_size_in_bytes	min_depth	max_depth	mean_depth	min_leaves	max_leaves	mean_leaves
0		10.0	20.0	9099.0	5.0	5.0	5.0	29.0	32.0	31.6


ModelMetricsBinomial: drf
** Reported on train data. **

MSE: 0.08916599897611499
RMSE: 0.298606763111814
LogLoss: 0.32003463451246766
Mean Per-Class Error: 0.42689792184224207
AUC: 0.6011218987561718
AUCPR: 0.1404187560183176
Gini: 0.20224379751234367

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.10328275148525204: 
0	1	Error	Rate
0	0	93574.0	61047.0	0.3948	(61047.0/154621.0)
1	1	7916.0	9318.0	0.4593	(7916.0/17234.0)
2	Total	101490.0	70365.0	0.4013	(68963.0/171855.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric	threshold	value	idx
0	max f1	0.103283	0.212742	248.0
1	max f2	0.075699	0.368283	336.0
2	max f0point5	0.137911	0.168827	147.0
3	max accuracy	0.389694	0.899724	3.0
4	max precision	0.389694	0.571429	3.0
5	max recall	0.025126	1.000000	399.0
6	max specificity	0.517241	0.999994	0.0
7	max absolute_mcc	0.111437	0.090045	221.0
8	max min_per_class_accuracy	0.100963	0.571475	255.0
9	max mean_per_class_accuracy	0.097455	0.573102	266.0
10	max tns	0.517241	154620.000000	0.0
11	max fns	0.517241	17233.000000	0.0
12	max fps	0.025126	154621.000000	399.0
13	max tps	0.025126	17234.000000	399.0
14	max tnr	0.517241	0.999994	0.0
15	max fnr	0.517241	0.999942	0.0
16	max fpr	0.025126	1.000000	399.0
17	max tpr	0.025126	1.000000	399.0

Gains/Lift Table: Avg response rate: 10.03 %, avg score:  9.91 %
group	cumulative_data_fraction	lower_threshold	lift	cumulative_lift	response_rate	score	cumulative_response_rate	cumulative_score	capture_rate	cumulative_capture_rate	gain	cumulative_gain	kolmogorov_smirnov
0	1	0.010004	0.187101	2.014981	2.014981	0.202073	0.215220	0.202073	0.215220	0.020157	0.020157	101.498107	101.498107	0.011285
1	2	0.020002	0.170442	2.096558	2.055758	0.210253	0.177787	0.206162	0.196509	0.020961	0.041119	109.655768	105.575763	0.023471
2	3	0.030005	0.161731	1.773872	1.961778	0.177893	0.165872	0.196737	0.186295	0.017745	0.058864	77.387222	96.177779	0.032075
3	4	0.040003	0.156086	1.769150	1.913635	0.177419	0.158969	0.191909	0.179465	0.017688	0.076552	76.915004	91.363472	0.040622
4	5	0.050001	0.151728	1.642782	1.859477	0.164747	0.153755	0.186478	0.174324	0.016425	0.092977	64.278218	85.947669	0.047765
5	6	0.100003	0.138445	1.473569	1.666523	0.147777	0.144368	0.167127	0.159346	0.073681	0.166657	47.356924	66.652296	0.074084
6	7	0.150004	0.130447	1.352973	1.562006	0.135683	0.134201	0.156646	0.150965	0.067651	0.234308	35.297316	56.200636	0.093700
7	8	0.200000	0.123805	1.273871	1.489979	0.127750	0.127040	0.149423	0.144984	0.063688	0.297996	27.387103	48.997875	0.108919
8	9	0.300003	0.112019	1.238120	1.406024	0.124165	0.117642	0.141003	0.135870	0.123816	0.421811	23.811975	40.602414	0.135386
9	10	0.400017	0.103499	1.068014	1.321513	0.107106	0.107448	0.132528	0.128764	0.106817	0.528628	6.801369	32.151301	0.142946
10	11	0.500003	0.096039	1.002269	1.257674	0.100513	0.099929	0.126126	0.122998	0.100212	0.628841	0.226916	25.767380	0.143198
11	12	0.600000	0.087915	0.911989	1.200061	0.091459	0.091917	0.120348	0.117818	0.091196	0.720037	-8.801141	20.006126	0.133416
12	13	0.700003	0.079760	0.817756	1.145445	0.082009	0.083844	0.114871	0.112964	0.081778	0.801815	-18.224373	14.544491	0.113160
13	14	0.800000	0.072046	0.752907	1.096379	0.075505	0.075788	0.109950	0.108317	0.075289	0.877103	-24.709254	9.637914	0.085697
14	15	0.900037	0.065681	0.618846	1.043302	0.062061	0.068690	0.104628	0.103913	0.061908	0.939011	-38.115396	4.330225	0.043318
15	16	1.000000	0.000000	0.610118	1.000000	0.061186	0.056225	0.100285	0.099146	0.060989	1.000000	-38.988244	0.000000	0.000000


ModelMetricsBinomial: drf
** Reported on validation data. **

MSE: 0.08806874434395472
RMSE: 0.2967637854320414
LogLoss: 0.31704941216882154
Mean Per-Class Error: 0.42692841456270203
AUC: 0.6005529428688353
AUCPR: 0.13815417010432543
Gini: 0.20110588573767063

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.10584315575891248: 
0	1	Error	Rate
0	0	24808.0	14216.0	0.3643	(14216.0/39024.0)
1	1	2099.0	2182.0	0.4903	(2099.0/4281.0)
2	Total	26907.0	16398.0	0.3767	(16315.0/43305.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric	threshold	value	idx
0	max f1	0.105843	0.211035	235.0
1	max f2	0.075094	0.365545	352.0
2	max f0point5	0.145585	0.169114	113.0
3	max accuracy	0.288220	0.901143	1.0
4	max precision	0.288220	0.500000	1.0
5	max recall	0.062025	1.000000	397.0
6	max specificity	0.303559	0.999974	0.0
7	max absolute_mcc	0.105843	0.089473	235.0
8	max min_per_class_accuracy	0.100562	0.568727	254.0
9	max mean_per_class_accuracy	0.089830	0.573072	296.0
10	max tns	0.303559	39023.000000	0.0
11	max fns	0.303559	4281.000000	0.0
12	max fps	0.059346	39024.000000	399.0
13	max tps	0.062025	4281.000000	397.0
14	max tnr	0.303559	0.999974	0.0
15	max fnr	0.303559	1.000000	0.0
16	max fpr	0.059346	1.000000	399.0
17	max tpr	0.062025	1.000000	397.0

Gains/Lift Table: Avg response rate:  9.89 %, avg score: 10.01 %
group	cumulative_data_fraction	lower_threshold	lift	cumulative_lift	response_rate	score	cumulative_response_rate	cumulative_score	capture_rate	cumulative_capture_rate	gain	cumulative_gain	kolmogorov_smirnov
0	1	0.010022	0.182614	1.981171	1.981171	0.195853	0.200785	0.195853	0.200785	0.019855	0.019855	98.117122	98.117122	0.010912
1	2	0.020021	0.166596	2.125917	2.053461	0.210162	0.173406	0.202999	0.187112	0.021257	0.041112	112.591703	105.346065	0.023405
2	3	0.030020	0.158153	1.775491	1.960875	0.175520	0.162065	0.193846	0.178769	0.017753	0.058865	77.549115	96.087542	0.032009
3	4	0.040018	0.153141	1.892300	1.943741	0.187067	0.155516	0.192152	0.172959	0.018921	0.077786	89.229977	94.374140	0.041910
4	5	0.050017	0.149086	2.125917	1.980160	0.210162	0.151093	0.195753	0.168588	0.021257	0.099042	112.591703	98.015971	0.054403
5	6	0.100012	0.137656	1.266206	1.623265	0.125173	0.142709	0.160471	0.155651	0.063303	0.162345	26.620553	62.326504	0.069172
6	7	0.150006	0.129997	1.298912	1.515164	0.128406	0.133564	0.149784	0.148290	0.064938	0.227283	29.891194	51.516399	0.085755
7	8	0.200000	0.122951	1.224154	1.442420	0.121016	0.126459	0.142593	0.142833	0.061201	0.288484	22.415442	44.242000	0.098191
8	9	0.300058	0.112402	1.272333	1.385702	0.125779	0.117347	0.136986	0.134334	0.127307	0.415791	27.233252	38.570235	0.128429
9	10	0.400092	0.103845	1.123180	1.320064	0.111034	0.108035	0.130498	0.127759	0.112357	0.528148	12.318021	32.006424	0.142103
10	11	0.500012	0.096150	1.000575	1.256220	0.098914	0.100004	0.124186	0.122212	0.099977	0.628124	0.057510	25.621953	0.142167
11	12	0.600647	0.088735	0.977210	1.209473	0.096604	0.092363	0.119565	0.117211	0.098342	0.726466	-2.279049	20.947294	0.139622
12	13	0.699988	0.081246	0.801820	1.151619	0.079265	0.085034	0.113846	0.112645	0.079654	0.806120	-19.818018	15.161909	0.117774
13	14	0.800000	0.073652	0.773095	1.104298	0.076426	0.077471	0.109168	0.108247	0.077318	0.883438	-22.690543	10.429806	0.092592
14	15	0.900058	0.067752	0.684025	1.057577	0.067621	0.070473	0.104549	0.104048	0.068442	0.951880	-31.597536	5.757705	0.057508
15	16	1.000000	0.059233	0.481474	1.000000	0.047597	0.064977	0.098857	0.100143	0.048120	1.000000	-51.852606	0.000000	0.000000


Scoring History: 
timestamp	duration	number_of_trees	training_rmse	training_logloss	training_auc	training_pr_auc	training_lift	training_classification_error	validation_rmse	validation_logloss	validation_auc	validation_pr_auc	validation_lift	validation_classification_error
0		2020-09-01 17:33:29	0.095 sec	0.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1		2020-09-01 17:33:32	2.689 sec	1.0	0.296994	0.317959	0.589940	0.135326	1.868401	0.476159	0.297120	0.318097	0.589347	0.134959	2.236086	0.430897
2		2020-09-01 17:33:34	4.232 sec	2.0	0.299017	0.321343	0.588942	0.134975	1.787957	0.446744	0.296871	0.317366	0.597396	0.136147	1.902779	0.381226
3		2020-09-01 17:33:41	11.612 sec	10.0	0.298607	0.320035	0.601122	0.140419	2.014981	0.401286	0.296764	0.317049	0.600553	0.138154	1.981171	0.376746

Variable Importances: 
variable	relative_importance	scaled_importance	percentage
0	ps_car_12 ps_car_15	429.612183	1.000000	0.140458
1	ps_car_13 ps_car_15	339.197632	0.789544	0.110898
2	ps_car_13^2	275.096161	0.640336	0.089940
3	ps_reg_01 ps_car_13	231.658600	0.539227	0.075739
4	ps_reg_02 ps_car_13	187.861847	0.437282	0.061420
5	ps_ind_17_bin	127.178253	0.296030	0.041580
6	ps_car_13	119.494781	0.278146	0.039068
7	ps_car_11_cat_te	114.261490	0.265964	0.037357
8	ps_reg_02	104.809006	0.243962	0.034266
9	ps_reg_03 ps_car_14	93.855797	0.218466	0.030685
10	ps_reg_02 ps_reg_03	69.528275	0.161840	0.022732
11	ps_ind_03	67.101814	0.156192	0.021938
12	ps_car_07_cat_1.0	61.672958	0.143555	0.020163
13	ps_reg_03 ps_car_13	57.897408	0.134767	0.018929
14	ps_reg_02 ps_car_15	55.194241	0.128475	0.018045
15	ps_reg_03	44.946861	0.104622	0.014695
16	ps_ind_05_cat_6.0	43.915367	0.102221	0.014358
17	ps_ind_16_bin	39.800083	0.092642	0.013012
18	ps_car_12 ps_car_13	38.566631	0.089771	0.012609
19	ps_reg_01 ps_car_15	37.650478	0.087638	0.012309

See the whole table with table.as_data_frame()
<bound method ModelBase.summary of >
```

##### Performance

```
ModelMetricsBinomial: drf
** Reported on train data. **

MSE: 0.08916599897611499
RMSE: 0.298606763111814
LogLoss: 0.32003463451246766
Mean Per-Class Error: 0.42689792184224207
AUC: 0.6011218987561718
AUCPR: 0.1404187560183176
Gini: 0.20224379751234367

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.10328275148525204: 
0	1	Error	Rate
0	0	93574.0	61047.0	0.3948	(61047.0/154621.0)
1	1	7916.0	9318.0	0.4593	(7916.0/17234.0)
2	Total	101490.0	70365.0	0.4013	(68963.0/171855.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric	threshold	value	idx
0	max f1	0.103283	0.212742	248.0
1	max f2	0.075699	0.368283	336.0
2	max f0point5	0.137911	0.168827	147.0
3	max accuracy	0.389694	0.899724	3.0
4	max precision	0.389694	0.571429	3.0
5	max recall	0.025126	1.000000	399.0
6	max specificity	0.517241	0.999994	0.0
7	max absolute_mcc	0.111437	0.090045	221.0
8	max min_per_class_accuracy	0.100963	0.571475	255.0
9	max mean_per_class_accuracy	0.097455	0.573102	266.0
10	max tns	0.517241	154620.000000	0.0
11	max fns	0.517241	17233.000000	0.0
12	max fps	0.025126	154621.000000	399.0
13	max tps	0.025126	17234.000000	399.0
14	max tnr	0.517241	0.999994	0.0
15	max fnr	0.517241	0.999942	0.0
16	max fpr	0.025126	1.000000	399.0
17	max tpr	0.025126	1.000000	399.0

Gains/Lift Table: Avg response rate: 10.03 %, avg score:  9.91 %
group	cumulative_data_fraction	lower_threshold	lift	cumulative_lift	response_rate	score	cumulative_response_rate	cumulative_score	capture_rate	cumulative_capture_rate	gain	cumulative_gain	kolmogorov_smirnov
0	1	0.010004	0.187101	2.014981	2.014981	0.202073	0.215220	0.202073	0.215220	0.020157	0.020157	101.498107	101.498107	0.011285
1	2	0.020002	0.170442	2.096558	2.055758	0.210253	0.177787	0.206162	0.196509	0.020961	0.041119	109.655768	105.575763	0.023471
2	3	0.030005	0.161731	1.773872	1.961778	0.177893	0.165872	0.196737	0.186295	0.017745	0.058864	77.387222	96.177779	0.032075
3	4	0.040003	0.156086	1.769150	1.913635	0.177419	0.158969	0.191909	0.179465	0.017688	0.076552	76.915004	91.363472	0.040622
4	5	0.050001	0.151728	1.642782	1.859477	0.164747	0.153755	0.186478	0.174324	0.016425	0.092977	64.278218	85.947669	0.047765
5	6	0.100003	0.138445	1.473569	1.666523	0.147777	0.144368	0.167127	0.159346	0.073681	0.166657	47.356924	66.652296	0.074084
6	7	0.150004	0.130447	1.352973	1.562006	0.135683	0.134201	0.156646	0.150965	0.067651	0.234308	35.297316	56.200636	0.093700
7	8	0.200000	0.123805	1.273871	1.489979	0.127750	0.127040	0.149423	0.144984	0.063688	0.297996	27.387103	48.997875	0.108919
8	9	0.300003	0.112019	1.238120	1.406024	0.124165	0.117642	0.141003	0.135870	0.123816	0.421811	23.811975	40.602414	0.135386
9	10	0.400017	0.103499	1.068014	1.321513	0.107106	0.107448	0.132528	0.128764	0.106817	0.528628	6.801369	32.151301	0.142946
10	11	0.500003	0.096039	1.002269	1.257674	0.100513	0.099929	0.126126	0.122998	0.100212	0.628841	0.226916	25.767380	0.143198
11	12	0.600000	0.087915	0.911989	1.200061	0.091459	0.091917	0.120348	0.117818	0.091196	0.720037	-8.801141	20.006126	0.133416
12	13	0.700003	0.079760	0.817756	1.145445	0.082009	0.083844	0.114871	0.112964	0.081778	0.801815	-18.224373	14.544491	0.113160
13	14	0.800000	0.072046	0.752907	1.096379	0.075505	0.075788	0.109950	0.108317	0.075289	0.877103	-24.709254	9.637914	0.085697
14	15	0.900037	0.065681	0.618846	1.043302	0.062061	0.068690	0.104628	0.103913	0.061908	0.939011	-38.115396	4.330225	0.043318
15	16	1.000000	0.000000	0.610118	1.000000	0.061186	0.056225	0.100285	0.099146	0.060989	1.000000	-38.988244	0.000000	0.000000
```

##### Prediction
```
predict	p0	p1	cal_p0	cal_p1
0	0.907694	0.0923055	0.912662	0.0873384
1	0.866307	0.133693	0.865248	0.134752
0	0.922823	0.0771773	0.925848	0.0741525
1	0.88674	0.11326	0.890904	0.109096
1	0.890054	0.109946	0.894637	0.105363
0	0.932941	0.0670592	0.933618	0.0663818
0	0.925728	0.0742717	0.928161	0.0718391
1	0.892583	0.107417	0.897409	0.102591
1	0.890709	0.109291	0.895361	0.104639
0	0.902714	0.0972858	0.907876	0.0921238
```

![download](https://user-images.githubusercontent.com/70437668/148633449-cb2f7b8e-ffce-4104-bc05-f665609c8282.png)

####  Gradient Boosting H2O

Reference: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html

Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method. The guiding heuristic is that good predictive results can be obtained through increasingly refined approximations. H2O’s GBM sequentially builds regression trees on all the features of the dataset in a fully distributed way - each tree is built in parallel.

The current version of GBM is fundamentally the same as in previous versions of H2O (same algorithmic steps, same histogramming techniques), with the exception of the following changes:

Improved ability to train on categorical variables (using the nbins_cats parameter)

Minor changes in histogramming logic for some corner cases

There was some code cleanup and refactoring to support the following features:

+ Per-row observation weights

+ Per-row offsets

+ N-fold cross-validation

+ Support for more distribution functions (such as Gamma, Poisson, and Tweedie)

##### Summary
```
Model Details
=============
H2OGradientBoostingEstimator :  Gradient Boosting Machine
Model Key:  GBM_model_python_1598981279148_10


Model Summary: 
number_of_trees	number_of_internal_trees	model_size_in_bytes	min_depth	max_depth	mean_depth	min_leaves	max_leaves	mean_leaves
0		50.0	50.0	22419.0	5.0	5.0	5.0	25.0	32.0	31.0


ModelMetricsBinomial: gbm
** Reported on train data. **

MSE: 0.39149036906036316
RMSE: 0.6256919122542365
LogLoss: 1.1408349597166443
Mean Per-Class Error: 0.37570927055088266
AUC: 0.6740573380548194
AUCPR: 0.6731829016300585
Gini: 0.34811467610963875

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.06597193734869071: 
0	1	Error	Rate
0	0	36524.0	119698.0	0.7662	(119698.0/156222.0)
1	1	13749.0	142501.0	0.088	(13749.0/156250.0)
2	Total	50273.0	262199.0	0.4271	(133447.0/312472.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric	threshold	value	idx
0	max f1	0.065972	0.681091	334.0
1	max f2	0.041907	0.834399	386.0
2	max f0point5	0.099651	0.624308	259.0
3	max accuracy	0.099651	0.624290	259.0
4	max precision	0.426954	1.000000	0.0
5	max recall	0.034675	1.000000	396.0
6	max specificity	0.426954	1.000000	0.0
7	max absolute_mcc	0.108140	0.249798	240.0
8	max min_per_class_accuracy	0.098214	0.620796	262.0
9	max mean_per_class_accuracy	0.099651	0.624291	259.0
10	max tns	0.426954	156222.000000	0.0
11	max fns	0.426954	156232.000000	0.0
12	max fps	0.024204	156222.000000	399.0
13	max tps	0.034675	156250.000000	396.0
14	max tnr	0.426954	1.000000	0.0
15	max fnr	0.426954	0.999885	0.0
16	max fpr	0.024204	1.000000	399.0
17	max tpr	0.034675	1.000000	396.0

Gains/Lift Table: Avg response rate: 50.00 %, avg score: 10.84 %
group	cumulative_data_fraction	lower_threshold	lift	cumulative_lift	response_rate	score	cumulative_response_rate	cumulative_score	capture_rate	cumulative_capture_rate	gain	cumulative_gain	kolmogorov_smirnov
0	1	0.010001	0.263911	1.836635	1.836635	0.918400	0.300426	0.918400	0.300426	0.018368	0.018368	83.663542	83.663542	0.016736
1	2	0.020021	0.237543	1.714315	1.775416	0.857234	0.248782	0.887788	0.274579	0.017178	0.035546	71.431460	77.541636	0.031052
2	3	0.030009	0.221763	1.640994	1.730676	0.820570	0.229095	0.865415	0.259440	0.016390	0.051936	64.099361	73.067567	0.043858
3	4	0.040000	0.210032	1.575131	1.691824	0.787636	0.215532	0.845988	0.248473	0.015738	0.067674	57.513112	69.182376	0.055351
4	5	0.050001	0.201028	1.576819	1.668821	0.788480	0.205330	0.834485	0.239844	0.015770	0.083443	57.681870	66.882127	0.066890
5	6	0.100019	0.171945	1.480832	1.574812	0.740482	0.184865	0.787476	0.212350	0.074067	0.157510	48.083218	57.481169	0.114994
6	7	0.150026	0.154350	1.396777	1.515468	0.698451	0.162565	0.757802	0.195756	0.069850	0.227360	39.677744	51.546820	0.154681
7	8	0.200021	0.141042	1.292292	1.459686	0.646204	0.147320	0.729908	0.183649	0.064608	0.291968	29.229234	45.968584	0.183910
8	9	0.300001	0.122263	1.199854	1.373093	0.599981	0.131015	0.686608	0.166108	0.119962	0.411930	19.985407	37.309281	0.223877
9	10	0.400001	0.108668	1.095047	1.303582	0.547573	0.115107	0.651849	0.153358	0.109504	0.521434	9.504701	30.358191	0.242888
10	11	0.500019	0.098143	1.024834	1.247824	0.512463	0.103201	0.623968	0.143325	0.102502	0.623936	2.483377	24.782408	0.247856
11	12	0.599999	0.088935	0.934201	1.195564	0.467143	0.093454	0.597836	0.135015	0.093402	0.717338	-6.579864	19.556394	0.234698
12	13	0.700015	0.079915	0.861628	1.147852	0.430852	0.084357	0.573978	0.127777	0.086176	0.803514	-13.837236	14.785243	0.207016
13	14	0.799998	0.070218	0.803334	1.104795	0.401703	0.075077	0.552447	0.121191	0.080320	0.883834	-19.666631	10.479465	0.167686
14	15	0.899997	0.058232	0.660804	1.055463	0.330432	0.064446	0.527779	0.114886	0.066080	0.949914	-33.919577	5.546256	0.099841
15	16	1.000000	0.019337	0.500851	1.000000	0.250448	0.049820	0.500045	0.108379	0.050086	1.000000	-49.914882	0.000000	0.000000


ModelMetricsBinomial: gbm
** Reported on validation data. **

MSE: 0.08722539145886551
RMSE: 0.29533945124020516
LogLoss: 0.3126858556635803
Mean Per-Class Error: 0.4050039427338912
AUC: 0.6312797740217534
AUCPR: 0.15752884275200094
Gini: 0.2625595480435068

Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.11816363929843933: 
0	1	Error	Rate
0	0	30387.0	8637.0	0.2213	(8637.0/39024.0)
1	1	2621.0	1660.0	0.6122	(2621.0/4281.0)
2	Total	33008.0	10297.0	0.26	(11258.0/43305.0)

Maximum Metrics: Maximum metrics at their respective thresholds
metric	threshold	value	idx
0	max f1	0.118164	0.227740	193.0
1	max f2	0.073967	0.376733	304.0
2	max f0point5	0.146817	0.189528	139.0
3	max accuracy	0.374071	0.901143	3.0
4	max precision	0.374071	0.500000	3.0
5	max recall	0.036831	1.000000	393.0
6	max specificity	0.397167	0.999974	0.0
7	max absolute_mcc	0.118164	0.116686	193.0
8	max min_per_class_accuracy	0.096178	0.591790	243.0
9	max mean_per_class_accuracy	0.098218	0.594996	238.0
10	max tns	0.397167	39023.000000	0.0
11	max fns	0.397167	4281.000000	0.0
12	max fps	0.023485	39024.000000	399.0
13	max tps	0.036831	4281.000000	393.0
14	max tnr	0.397167	0.999974	0.0
15	max fnr	0.397167	1.000000	0.0
16	max fpr	0.023485	1.000000	399.0
17	max tpr	0.036831	1.000000	393.0

Gains/Lift Table: Avg response rate:  9.89 %, avg score:  9.71 %
group	cumulative_data_fraction	lower_threshold	lift	cumulative_lift	response_rate	score	cumulative_response_rate	cumulative_score	capture_rate	cumulative_capture_rate	gain	cumulative_gain	kolmogorov_smirnov
0	1	0.010022	0.227989	2.773640	2.773640	0.274194	0.258511	0.274194	0.258511	0.027797	0.027797	177.363971	177.363971	0.019725
1	2	0.020021	0.206114	2.499705	2.636830	0.247113	0.215343	0.260669	0.236952	0.024994	0.052791	149.970464	163.683016	0.036366
2	3	0.030020	0.192172	2.196002	2.490001	0.217090	0.198556	0.246154	0.224163	0.021957	0.074749	119.600221	149.000054	0.049636
3	4	0.040018	0.182788	1.775491	2.311476	0.175520	0.187088	0.228505	0.214900	0.017753	0.092502	77.549115	131.147626	0.058241
4	5	0.050017	0.175022	1.705406	2.190318	0.168591	0.178887	0.216528	0.207700	0.017052	0.109554	70.540597	119.031817	0.066068
5	6	0.100012	0.150185	1.616631	1.903541	0.159815	0.161315	0.188178	0.184513	0.080822	0.190376	61.663141	90.354102	0.100278
6	7	0.150006	0.134732	1.481133	1.762760	0.146420	0.142005	0.174261	0.170346	0.074048	0.264424	48.113340	76.276016	0.126970
7	8	0.200000	0.124303	1.392359	1.670171	0.137644	0.129290	0.165108	0.160083	0.069610	0.334034	39.235885	67.017052	0.148738
8	9	0.300012	0.109104	1.174823	1.505042	0.116139	0.116239	0.148784	0.145467	0.117496	0.451530	17.482348	50.504213	0.168140
9	10	0.400000	0.098509	1.165750	1.420229	0.115242	0.103492	0.140399	0.134975	0.116562	0.568092	16.575011	42.022892	0.186532
10	11	0.500012	0.089375	0.957610	1.327697	0.094666	0.093859	0.131252	0.126751	0.095772	0.663864	-4.239041	32.769651	0.181827
11	12	0.600000	0.081398	0.885409	1.253991	0.087529	0.085353	0.123966	0.119852	0.088531	0.752394	-11.459060	25.399050	0.169112
12	13	0.699988	0.072982	0.808316	1.190329	0.079908	0.077247	0.117672	0.113766	0.080822	0.833217	-19.168429	19.032897	0.147843
13	14	0.800000	0.064104	0.705361	1.129701	0.069730	0.068584	0.111679	0.108118	0.070544	0.903761	-29.463879	12.970100	0.115144
14	15	0.899988	0.053792	0.567690	1.067262	0.056120	0.059094	0.105506	0.102671	0.056762	0.960523	-43.231007	6.726174	0.067176
15	16	1.000000	0.022663	0.394722	1.000000	0.039021	0.046819	0.098857	0.097086	0.039477	1.000000	-60.527800	0.000000	0.000000


Scoring History: 
timestamp	duration	number_of_trees	training_rmse	training_logloss	training_auc	training_pr_auc	training_lift	training_classification_error	validation_rmse	validation_logloss	validation_auc	validation_pr_auc	validation_lift	validation_classification_error
0		2020-09-01 17:45:31	0.025 sec	0.0	0.640162	1.202806	0.500000	0.500045	1.000000	0.499955	0.298473	0.322575	0.500000	0.098857	1.000000	0.901143
1		2020-09-01 17:46:03	32.217 sec	7.0	0.636285	1.181877	0.633946	0.629378	1.696876	0.468448	0.296887	0.317453	0.612053	0.146672	2.528907	0.326544
2		2020-09-01 17:46:13	42.337 sec	10.0	0.634891	1.175765	0.639630	0.634004	1.689064	0.458105	0.296541	0.316341	0.615629	0.147727	2.551486	0.366886
3		2020-09-01 17:46:20	49.048 sec	12.0	0.634054	1.172286	0.643137	0.637518	1.692747	0.451058	0.296319	0.315652	0.619155	0.150163	2.727024	0.371874
4		2020-09-01 17:46:26	55.920 sec	14.0	0.633309	1.169300	0.645837	0.640412	1.708380	0.444859	0.296146	0.315147	0.620839	0.152392	2.843563	0.354370
5		2020-09-01 17:46:33	1 min 2.557 sec	16.0	0.632599	1.166492	0.648422	0.643709	1.743844	0.440612	0.295999	0.314694	0.622954	0.153766	2.794500	0.340838
6		2020-09-01 17:46:43	1 min 12.406 sec	19.0	0.631643	1.163025	0.650628	0.647685	1.746250	0.439883	0.295871	0.314307	0.623946	0.153893	2.796948	0.351553
7		2020-09-01 17:46:50	1 min 19.246 sec	21.0	0.631031	1.160788	0.652424	0.649791	1.749604	0.455980	0.295782	0.314036	0.625077	0.154496	2.790518	0.301097
8		2020-09-01 17:46:56	1 min 25.797 sec	23.0	0.630466	1.158679	0.654573	0.652014	1.767967	0.438250	0.295714	0.313814	0.625946	0.154704	2.727024	0.315414
9		2020-09-01 17:47:03	1 min 32.241 sec	25.0	0.629982	1.156845	0.656536	0.654391	1.780173	0.442785	0.295671	0.313691	0.626249	0.155284	2.820256	0.325644
10		2020-09-01 17:47:10	1 min 39.117 sec	27.0	0.629481	1.155179	0.657899	0.656249	1.789347	0.445326	0.295621	0.313552	0.626656	0.155227	2.890179	0.336428
11		2020-09-01 17:47:16	1 min 45.760 sec	29.0	0.629006	1.153469	0.659438	0.657961	1.794465	0.441569	0.295581	0.313435	0.627433	0.155523	2.727024	0.306431
12		2020-09-01 17:47:24	1 min 53.834 sec	31.0	0.628648	1.152017	0.661265	0.659529	1.800286	0.436532	0.295569	0.313393	0.627533	0.155516	2.703716	0.293292
13		2020-09-01 17:47:34	2 min 3.648 sec	34.0	0.628069	1.149896	0.663629	0.661962	1.806620	0.428384	0.295512	0.313195	0.628973	0.155569	2.727024	0.282785
14		2020-09-01 17:47:44	2 min 13.481 sec	37.0	0.627596	1.147963	0.665988	0.664083	1.820980	0.424841	0.295473	0.313071	0.629511	0.155915	2.727024	0.275257
15		2020-09-01 17:47:50	2 min 19.922 sec	39.0	0.627274	1.146769	0.667184	0.665621	1.824477	0.426637	0.295412	0.312913	0.630268	0.156924	2.750332	0.265073
16		2020-09-01 17:47:57	2 min 26.374 sec	41.0	0.626901	1.145447	0.668595	0.666997	1.825116	0.430679	0.295389	0.312824	0.630984	0.156930	2.727024	0.260455
17		2020-09-01 17:48:04	2 min 33.002 sec	43.0	0.626638	1.144448	0.669767	0.668300	1.827367	0.422211	0.295370	0.312772	0.631092	0.157186	2.773640	0.274887
18		2020-09-01 17:48:10	2 min 39.581 sec	45.0	0.626328	1.143381	0.670728	0.669708	1.834288	0.422486	0.295353	0.312737	0.631048	0.157509	2.703716	0.272070
19		2020-09-01 17:48:17	2 min 46.209 sec	47.0	0.626037	1.142217	0.672346	0.671257	1.833116	0.428358	0.295329	0.312667	0.631419	0.157678	2.820256	0.273710

See the whole table with table.as_data_frame()

Variable Importances: 
variable	relative_importance	scaled_importance	percentage
0	ps_car_11_cat_te	3484.061279	1.000000	0.096238
1	ps_ind_03	2929.698730	0.840886	0.080925
2	ps_ind_17_bin	2428.832520	0.697127	0.067090
3	ps_car_13	2117.113037	0.607657	0.058480
4	ps_ind_15	1460.435059	0.419176	0.040341
5	ps_car_13^2	1291.965698	0.370822	0.035687
6	ps_ind_05_cat_6.0	1235.729492	0.354681	0.034134
7	ps_car_07_cat_1.0	1200.827881	0.344663	0.033170
8	ps_reg_02 ps_car_13	1146.297852	0.329012	0.031664
9	ps_reg_03 ps_car_13	1085.685913	0.311615	0.029989
10	ps_reg_02 ps_car_15	951.527954	0.273109	0.026283
11	ps_ind_06_bin	939.708923	0.269717	0.025957
12	ps_reg_01 ps_car_13	915.752136	0.262840	0.025295
13	ps_reg_01 ps_reg_03	772.698486	0.221781	0.021344
14	ps_ind_05_cat_4.0	718.870972	0.206331	0.019857
15	ps_reg_01 ps_car_15	709.557678	0.203658	0.019600
16	ps_ind_05_cat_2.0	620.422241	0.178074	0.017138
17	ps_ind_16_bin	545.809265	0.156659	0.015077
18	ps_car_12 ps_car_13	512.391357	0.147067	0.014153
19	ps_ind_01	473.627319	0.135941	0.013083

See the whole table with table.as_data_frame()
```

##### Prediction
```
predict	p0	p1
0	0.892602	0.107398
0	0.909249	0.0907512
1	0.868848	0.131152
0	0.91712	0.0828795
0	0.955331	0.0446686
0	0.926904	0.0730959
0	0.912628	0.087372
0	0.927763	0.0722372
0	0.884861	0.115139
0	0.882893	0.117107
```

![download (1)](https://user-images.githubusercontent.com/70437668/148633591-e8b45487-a2b3-469a-871d-c97d878f4fc2.png)


### C. Feature Importance

#### Decision Tree

![Decision Tree Feature Importance](https://user-images.githubusercontent.com/70437668/141063370-2436a59f-680b-455d-bb2f-9913021f6a70.jpg)

#### Random Forest 

![Random Forest Feature Importance](https://user-images.githubusercontent.com/70437668/141063361-a29a1527-fb9d-4d36-8642-91d36c2f18a7.jpg)

#### Balanced Random Forest

![Balanced Random Forest Feature Importance](https://user-images.githubusercontent.com/70437668/141063356-b7168576-5acd-4242-bc91-49d9e8f4b46c.jpg)

#### RUS Boost

![RUS Boost Feature Importance](https://user-images.githubusercontent.com/70437668/141063346-b28aec1b-9463-47f8-88b5-1d0a61622b7f.jpg)

#### XGBoost

![XGBoost Feature Importance](https://user-images.githubusercontent.com/70437668/141063338-e60ea688-48f9-4df0-8d13-5110c664b978.jpg)


### D. Heatmap

##### Train set

<img src="https://user-images.githubusercontent.com/70437668/141063258-0d1b0dec-d7e5-4bc3-a5f2-8f232aff86da.jpg" width=50% height=50%>

##### Test set

<img src="https://user-images.githubusercontent.com/70437668/141063268-814e866d-7ad9-4fe5-acc3-89f2635d0ff9.jpg" width=50% height=50%>

### E. Draw Single Decision Tree

![download](https://user-images.githubusercontent.com/70437668/141065326-d2dd2570-b789-4a08-8aa5-7ccf4f2cbdd4.png)

### F. ROC & AUC between Deep Neural Network, Ensemble Classifiers, XGBoost Classifier

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

### G. Predict

![Predict](https://user-images.githubusercontent.com/70437668/141063151-3e227c9e-0a2b-4c0c-ba4f-97442010813a.jpg)

### H. New Policy on Trial:

#### H.1 List out

#### H.2 Implement that New Policy
```
result = dectree.predict(new_policy)
len(np.where(result==1)[0])
```

#### Output
```
30537
```
#### H.3 Result

**30537 drivers will claim insurance instead of 3 with entropy = 0 when changing ps_car_13 (most influential feature by Single Decision Tree with max_depth=5) to 2.5 for example (as long as greater than 2.447).**

## References:

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

[12] https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html

[13] https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html
