# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:01:18 2019

@author: CBIR Team
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
import pickle

dataset = pd.read_excel("HybridFeatureMyDB1_1.xlsx",header=None, index=None)
responses = pd.read_excel("Responses.xlsx",header=None, index=None)

X = dataset.iloc[:,:].values
conv = responses.values
y = label_binarize(conv,classes=[0,1,2,3,4,5,6,7,8,9])

n_classes = y.shape[1] # return number of class

random_state = np.random.RandomState(24)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test) 

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
classifier = OneVsRestClassifier(svm.LineaclsrSVC(random_state=0, max_iter=5000))
classifier.fit(X_train, y_train)


with open('SVM.pickle','wb') as f:
    pickle.dump(classifier,f)
y_score = classifier.decision_function(X_test)
print(y_score)
y_pred = classifier.predict(X_test)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

from sklearn.utils.fixes import signature
step_kwargs = ({'step': 'post'}
                 if 'step' in signature(plt.fill_between).parameters 
                 else {})
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
target_names = ['0','1','2','3','4','5','6','7','8','9']
print(classification_report(y_test, y_pred, target_names=target_names))
#confusion_matrix(y_test,y_pred)
