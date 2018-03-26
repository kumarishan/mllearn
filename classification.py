import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_mldata
import numpy as np

# %%
mnist = fetch_mldata('MNIST original')
mnist

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

# %%
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")

plt.show()

# %%

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

# %%
plt.figure(figsize=(9,9))
# concatenate in first dimension and returns array
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# %%
"""
# Binary Classifier
"""
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])


# %%
"""
Model selection
"""
from sklearn.model_selection import cross_val_score
# cv=3 means it uses StratifiedKFold - The folds are made by preserving the percentage of samples for each class
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %%
"""
Code to replicate the above one line cross validation
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_fold = X_train[train_index]
    y_train_fold = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_fold, y_train_fold)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# %%
"""
Get cross validated score of input
returns for each element in the input, the prediction that was obtained for that element when it was in the test set.
"""
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=5)

# %%
"""
Confusion Matrix
Compute confusion matrix to evaluate the accuracy of classification
each entry i, j is number of observation in group i predicted to be in group j

- “true positive” for correctly predicted yes values.
- “false positive” for incorrectly predicted no values.
- “true negative” for correctly predicted no values.
- “false negative” for incorrectly predicted yes values.

Confustion Matrix of 2-class problems
  		no			        yes
no   	true negative		false positive
yes	    false negative		true positive


Accuracy = (TP + TN) / Total
Missclassification rate/Error rate = (FP + FN) / Total - How often it is wrong?
True positive rate/Recall/Sensitivity = TP / (TP + FN) - When it's actually yes, how often does it predict yes?
False Positive Rate = FP / (TN + FP) = 1 - Specificity - When it's actually no, how often does it predict yes?
Specificity = TN / (TN + FP) - When it's actually no, how often does it predict no?
Precision = TP / (FP + TP) When it predicts yes, how often is it correct?
Prevalence (TP + FN) / Total - How often does the yes condition actually occur in our sample?

"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
# %%
"""
Precision
Precision = tp / (tp + fp)
The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
The best value is 1 and the worst value is 0.
"""
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
# This is equal to
tp = cm[1][1] # true positives are those that are 5 and classified as five
fp = cm[0][1] # false positives are those that are not 5 and classfied as five
precision = tp / (tp + fp)
print(precision)

# %%
"""
Recall
Recall = tp / (tp + fn)
fn - which wrongly indicates that a particular condition or attribute is absent.
The recall is intuitively the ability of the classifier to find all the positive samples.
"""
recall_score(y_train_5, y_train_pred)
tp = cm[1][1]
fn = cm[1][0]
recall = tp / (tp + tn)
print(recall)

# %%
"""
F1 score
is the harmonic mean of precision and recall
F1_score = 2 / (1/precision + 1/recall)
why harmonic mean? regular mean treats all values equally, but harmonic mean gives much more weight to low values
therefore a classification will only get high score if both precision and recall are high
F1 favors model that have similar precision and recall

measures the effectiveness of retrieval with respect to a user who attaches β times as much importance to recall as precision

Precision/Recall tradeoff
"""
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# %%
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# y_scores contains the scores computed using the method=decision_function

# This is due to a bug in sklearn 19.0
if y_scores.ndim == 2:
    y_scores = y_scores[:, 1]

# %%
"""
Precision/Recall tradeoff -  increasing precision reduces recall, and vice versa
Precision recall curve
"""

# Returns a score for each instance
y_some_scores = sgd_clf.decision_function([some_digit])

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
plt.show()

# %%
"""
Precision/Recall curve
"""

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plot_precision_vs_recall(precisions, recalls)
plt.show()

# %%
"""
ROC curve - Receiver Operating Characteristic
true positive rate vs false positive rate
Sensitivity (recall) vs 1 - Specificity
"""
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label) # plot simple line
    plt.plot([0, 1], [0, 1], 'k--') # plot straight line
    plt.axis([0, 1, 0, 1]) # axis ranges
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

# %%
"""
Tradeoff - the higher the recall (TPR), the more false positives
(FPR) the classifier produces.

The dottesd line represent the roc curve of the purely random classifier
A good classifier stays as far away from the line - towards top left corner

One way to compare classifier is to measure AUC (area under the curve)
A perfect classifier will have area = 1

Which curve to user Precision/Recall or ROC?
As a rule of thumb - when positive classes are rare or u care more about the false
positives than the false negatives - then use PR curve
Otherwise ROC curve
"""

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

# %%
"""
A exmaple with RandomForestClassifier
"""
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# To get scores from pobabilities for the ROC roc_curve
# is to use positive class probability as a score
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
roc_auc_score(y_train_5, y_scores_forest)
plot_roc_curve(fpr_forest, tpr_forest, thresholds_forest)
plt.show()

# %%
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
plot_precision_vs_recall(precisions_forest, recalls_forest)
plt.show()

# %%
plot_precision_recall_vs_threshold(precisions_forest, recalls_forest, thresholds_forest)

"""
can easily find precision = 98.5% and recall = 82.8%
"""

# %%
"""
# Multiclass classification

One-vs-All strategy OvA
One-vs-One strategy OvO - N(N-1)/2 classifiers
- benefit that each classifier needs to run on only part of the dataset
- some algorithm scale poorly with the size of the dataset - like SVM -
  so for these OvO is preferred as multiple can be trained over small dataset

SGDClassifier trains N binary classifiers
and decision_function returns the scores for each classfier and highest score
is chosen
"""
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

# %%
"""
RandomForestClassifier directly classify instances into multiple classes, therefore
predict_proba() to get a list of probabilities
"""
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# %%
"""
Error analysis
"""
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, y_train_pred)
cm

plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

# %%
"""
Normalize the confusion matrix
"""

row_sums = cm.sum(axis=1, keepdims=True)
norm_cm = cm / row_sums

# Remove the diagnols to keep only the errors
np.fill_diagonal(norm_cm, 0)
plt.matshow(norm_cm, cmap=plt.cm.gray)
plt.show()

# %%
"""
Multilable classification

KNeighborsClassifier

f1_score -- assumes all labels are equally important and hence simple average
other option average="weighted" -- to give each label a weight equal to its support (i.e., the number of instances with that
target label)
"""
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_train, y_train_knn_pred, average="macro")
