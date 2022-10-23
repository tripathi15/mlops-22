# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause
from statistics import median
# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
from skimage import transform
from tabulate import tabulate
###############################################################################
# Digits dataset
# --------------
def new_data(data,size):
	new_features = np.array(list(map(lambda img: transform.resize(
				img.reshape(8,8),(size,size),mode='constant',preserve_range=True).ravel(),data)))
	return new_features

digits = datasets.load_digits()
###############################################################################
# Classification
# --------------
user_split = 0.5
# flatten the images
n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
user_size = 8
data = new_data(digits.data,user_size)
print(" ")
print('For Image Size = '+str(user_size)+'x'+str(user_size)+' and Train-Val-Test Split => '+str(int(100*(1-user_split)))+
	'-'+str(int(50*user_split))+'-'+str(int(50*user_split)))

GAMMA = [10,1,0.1]
C = [1,2]

best_gam = 0
best_c = 0
best_mean_acc=0
best_train=0
best_val=0
best_test=0
table = [['Gamma','C','Training Acc.','Val (Dev) Acc.','Test Acc.','Min Acc.','Max Acc.','Median Acc.','Mean Acc.']]
for GAM in GAMMA:
	for c in C:
		hyper_params = {'gamma':GAM, 'C':c}
		clf = svm.SVC()
		clf.set_params(**hyper_params)
		X_train, X, y_train, y = train_test_split(data, digits.target, test_size=user_split, shuffle=False)
		x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
		clf.fit(X_train, y_train)
		predicted_val = clf.predict(x_val)
		predicted_train = clf.predict(X_train)
		predicted_test = clf.predict(x_test)
		accuracy_val = 100*metrics.accuracy_score(y_val,predicted_val)
		accuracy_train = 100*metrics.accuracy_score(y_train, predicted_train)
		accuracy_test = 100*metrics.accuracy_score(y_test, predicted_test)
		mean_acc = (accuracy_val + accuracy_train + accuracy_test)/3
		min_acc = min([accuracy_train,accuracy_val,accuracy_test])
		max_acc = max([accuracy_val,accuracy_train,accuracy_test])
		median_acc = median([accuracy_val,accuracy_train,accuracy_test])
		table.append([GAM,c,str(accuracy_train)+'%',str(accuracy_val)+'%',str(accuracy_test)+'%',str(min_acc)+'%',
				str(max_acc)+'%',str(median_acc)+'%',str(mean_acc)+'%'])
		if accuracy_test>best_test:
			best_gam = GAM
			best_c = c
			best_train=accuracy_train
			best_val=accuracy_val
			best_test=accuracy_test
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print(" ")
print('Best Hyperparameters (Gamma and C) => '+str(best_gam)+' and '+str(best_c))
print('Train, Val (Dev) and Test Accuracies => '+str(best_train)+'%, '+str(best_val)+'%, '+str(best_test)+'%')
print(" ")

max_features = ['sqrt','log2']
ccp_val = [0.01,0.1,1]

best_max = 0
best_ccp = 0
best_mean_acc = 0
best_train = 0
best_val = 0
best_test = 0

table_ii = [['max_features','ccp','Train_acc','Dev_Acc','Test_Acc','Min_Acc','Max_Acc','Median_Acc','Mean_Acc']]
for feat in max_features:
	for ccp in ccp_val:
		hyper_params = {'ccp_alpha':ccp, 'max_features':feat}
		clf = tree.DecisionTreeClassifier()
		clf.set_params(**hyper_params)
		X_train, X, y_train, y = train_test_split(data, digits.target, test_size=user_split, shuffle=False)
		x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
		clf.fit(X_train, y_train)
		predicted_val = clf.predict(x_val)
		predicted_train = clf.predict(X_train)
		predicted_test = clf.predict(x_test)
		accuracy_val = 100*metrics.accuracy_score(y_val,predicted_val)
		accuracy_train = 100*metrics.accuracy_score(y_train, predicted_train)
		accuracy_test = 100*metrics.accuracy_score(y_test, predicted_test)
		mean_acc = (accuracy_val + accuracy_train + accuracy_test)/3
		min_acc = min([accuracy_train,accuracy_val,accuracy_test])
		max_acc = max([accuracy_val,accuracy_train,accuracy_test])
		median_acc = median([accuracy_val,accuracy_train,accuracy_test])
		table_ii.append([feat,ccp,str(accuracy_train)+'%',str(accuracy_val)+'%',str(accuracy_test)+'%',str(min_acc)+'%',
				str(max_acc)+'%',str(median_acc)+'%',str(mean_acc)+'%'])
		if accuracy_test>best_test:
			best_max = feat
			best_ccp = ccp
			best_train=accuracy_train
			best_val=accuracy_val
			best_test=accuracy_test
print(tabulate(table_ii, headers='firstrow', tablefmt='fancy_grid'))
print(" ")
print('Best Hyperparameters (Max_Features and CCP_Alpha) => '+str(best_max)+' and '+str(best_ccp))
print('Train, Val (Dev) and Test Accuracies => '+str(best_train)+'%, '+str(best_val)+'%, '+str(best_test)+'%')
print(" ")
###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.
'''
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
'''
###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
'''
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
'''
###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.
'''
cm = metrics.confusion_matrix(y_test, predicted)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp = disp.plot()
'''
#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")
'''
plt.show()
'''
Footer