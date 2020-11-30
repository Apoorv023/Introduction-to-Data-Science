from sklearn.model_selection import train_test_split		# For splitting the dataset into train and test data
from sklearn.tree import DecisionTreeClassifier			# For implementing decision tree
from sklearn import tree 					# For visualizing decision tree
from sklearn.neighbors import KNeighborsClassifier		# For implementing KNN
from sklearn import naive_bayes					# For implementing Naive Bayes
from sklearn.naive_bayes import GaussianNB			# For implementing Naive Bayes
from sklearn import svm 					# For implementing Support Vector Machine
from sklearn.svm import SVC 					# For implementing Support Vector Machine
from sklearn.linear_model import LogisticRegression 		# For implementing Logistic Regression
from sklearn.metrics import confusion_matrix			# For creating confusion matrix
from sklearn.metrics import accuracy_score			# For checking the accuracy
from sklearn.metrics import classification_report		# For precision, recall, and f-score values
from sklearn.model_selection import KFold			# For dividing he dataset into 'k' parts(here, k = 10) 
from sklearn.model_selection import cross_val_score		# For implementing cross validation
import pandas as pd 						# For using dataframes to store the dataset and perform computations
import numpy as np 						# For using numpy library
import matplotlib.pyplot as plt 				# For plotting graphs
import seaborn							# For plotting pairplot and heatmap

# Reading the dataset from the file 'car.data' in the dataframe 'df'
df = pd.read_csv("car.data", sep=",", header = None)

# Data Pre-processing
df = df.replace(to_replace = ["vhigh", "vgood"], value = 4)														# Replacing the words 'vhigh' and 'vgood' in the dataset with number '4'
df = df.replace(to_replace = ["high", "big", "good"], value = 3)													# Replacing the words 'high', 'big' and 'good' in the dataset with number '3'
df = df.replace(to_replace = ["low", "small", "unacc"], value = 1)													# Replacing the words 'low', 'small' and 'unacc' in the dataset with number '1'
df = df.replace(to_replace = ["med", "acc"], value = 2)															# Replacing the words 'med' and 'acc' in the dataset with number '2'
df = df.replace(to_replace = ["5more", "more"], value = 5)														# Replacing the words '5more' and 'more' in the dataset with number '5'
df.columns = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'CLASS']											# Naming the columns of the dataframe 'df'
df['Doors'] = df['Doors'].astype(int)																	# Converting values from string datatype to int datatype in column 'Doors' in dataframe 'df'
df['Persons'] = df['Persons'].astype(int)																# Converting values from string datatype to int datatype in column 'Persons' in dataframe 'df'
print("Checking for Null values:\n", df.isnull().sum(), sep = "")													# Checking whether there are any Null value(s) present in the dataset
print("\nChecking for missing values:\n", df.isna().sum(), "\n\n", sep = "")												# Checking whether there are any missing value(s) present in the dataset
# Dividing the dataset into attributes(X) and Class(Y)
X = df.values[:, 0:6] 
Y = df.values[:, 6]
# Dividing the dataset into training data and test data in a 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)

# Plotting a pair plot
seaborn.pairplot(df, hue = 'CLASS')																	# Creating pair plot
plt.savefig('Pair_plot.png')																		# Saving above diagram in a png file called 'Pair_plot.png'
plt.close()
# Plotting a heatmap
plt.figure(figsize = (12,12))
correlation = df.corr()																			# Generating correlation matrix
print("\n\nCorrelation Matrix:\n\n", correlation, "\n", sep="")
seaborn.heatmap(correlation, annot = True)																# Creating heatmap
plt.savefig('Heatmap.png')																		# Saving above diagram in a png file called 'Heatmap.png'
plt.close()

# Generating bar graph for all the attributes and class present in the dataframe
# Generating bar graph for CLASS values
uniq_cls = list(set(Y))																			# 'uniq_cls' contains all the different values that are available for 'CLASS'
uniq_cls.sort()
frq_list = [0]*len(uniq_cls)																		# 'frq_list' will contain the frequency of each different value available for 'CLASS'. Here it is initialized with 0
# Updating 'frq_list' by iterating over the class values
for i in list(Y):
	frq_list[int(i)-1] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different CLASS values")
plt.ylabel("Frequency")
plt.savefig('Class_frequency.png')
plt.close()

# Generating bar graph for 'Buying' values(Attribute - 1)
y = df.values[:,0].tolist()
uniq_cls = list(set(y))
frq_list = [0]*len(uniq_cls)
for i in y:
	frq_list[int(i)-1] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different Buying(Attribute - 1) values")
plt.ylabel("Frequency")
plt.savefig('Buying_frequency.png')
plt.close()

# Generating bar graph for 'Miant' values(Attribute - 2)
y = df.values[:,1].tolist()
uniq_cls = list(set(y))
uniq_cls.sort()
frq_list = [0]*len(uniq_cls)
for i in y:
	frq_list[int(i)-1] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different Maint(Attribute - 2) values")
plt.ylabel("Frequency")
plt.savefig('Maint_frequency.png')
plt.close()

# Generating bar graph for 'Doors' values(Attribute - 3)
y = df.values[:,2].tolist()
uniq_cls = list(set(y))
uniq_cls.sort()
frq_list = [0]*len(uniq_cls)
for i in y:
	frq_list[int(i)-2] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different Doors(Attribute - 3) values")
plt.ylabel("Frequency")
plt.savefig('Doors_frequency.png')
plt.close()

# Generating bar graph for 'Persons' values(Attribute - 4)
y = df.values[:,3].tolist()
uniq_cls = list(set(y))
uniq_cls.sort()
frq_list = [0]*len(uniq_cls)
for i in y:
	if i == 2:
		frq_list[0] += 1
	elif i == 4:
		frq_list[1] += 1
	else:
		frq_list[2] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different Persons(Attribute - 4) values")
plt.ylabel("Frequency")
plt.savefig('Persons_frequency.png')
plt.close()

# Generating bar graph for 'Lug_boot' values(Attribute - 5)
y = df.values[:,4].tolist()
uniq_cls = list(set(y))
uniq_cls.sort()
frq_list = [0]*len(uniq_cls)
for i in y:
	frq_list[int(i)-1] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different Lug_boot(Attribute - 5) values")
plt.ylabel("Frequency")
plt.savefig('Lug_boot_frequency.png')
plt.close()

# Generating bar graph for 'Safety' values(Attribute - 6)
y = df.values[:,5].tolist()
uniq_cls = list(set(y))
uniq_cls.sort()
frq_list = [0]*len(uniq_cls)
for i in y:
	frq_list[int(i)-1] += 1
plt.bar(uniq_cls, frq_list)
plt.xlabel("Different Safety(Attribute - 6) values")
plt.ylabel("Frequency")
plt.savefig('Safety_frequency.png')
plt.close()

# Implementation of Decision Tree Algorithm using Gini index
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)								# Creating the classifier object
clf_gini.fit(X_train, y_train)																		# Performing training
y_pred = clf_gini.predict(X_test)																	# Predicting values for the test data 'X_test'
print("\n\t\tRESULT FOR DECISION TREE ALGORITHM USING GINI INDEX")
print("\nAccuracy for Decision Tree Algorithm using Gini index(in %):", accuracy_score(y_test,y_pred)*100) 								# Accuracy of the decision tree
print("Confusion Matrix for Decision Tree Algorithm using Gini index :\n", confusion_matrix(y_test, y_pred))								# Printing confusion matrix
print("\n", classification_report(y_test, y_pred, zero_division = 0))													# Printing precision, recall, and f-score
fig = plt.figure()
_ = tree.plot_tree(clf_gini, feature_names = df.columns[0:6], class_names = df.columns[6], filled = True)								# Creating decision tree
fig.savefig("decision_tree_gini.png")																	# Saving above diagram in a png file called 'decistion_tree_gini.png'


# Implementation of Decision Tree Algorithm using Entropy
clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)							# Creating the classifier object
clf_entropy.fit(X_train, y_train)																	# Performing training
y_pred = clf_entropy.predict(X_test)																	# Predicting values for the test data 'X_test'
print("\n\t\tRESULT FOR DECISION TREE ALGORITHM USING ENTROPY")
print("\nAccuracy for Decision Tree Algorithm using Entropy(in %):", accuracy_score(y_test,y_pred)*100)									# Accuracy of the decision tree
print("Confusion Matrix for Decision Tree Algorithm using Entropy :\n", confusion_matrix(y_test, y_pred))								# Printing confusion matrix
print("\n", classification_report(y_test, y_pred, zero_division = 0))													# Printing precision, recall, and f-score
fig = plt.figure()
_ = tree.plot_tree(clf_entropy, feature_names = df.columns[0:6], class_names = df.columns[6], filled=True)								# Creating decision tree
fig.savefig("decision_tree_entropy.png")																# Saving above diagram in a png file called 'decistion_tree_entropy.png'


# Implementation of KNN Algorithm for different values of 'K'
neighbors = np.arange(1, 11)																		# The range of values of 'K' for which the KNN algorithm will run. Here, this range is from [1,10]
train_accuracy = [0]*len(neighbors)																	# Initialising list 'train_accuracy' with '0' that will later contain the training data accuracy of for K = i+1, where 'i' is the index value of the list 'train_accuracy'
test_accuracy = [0]*len(neighbors)																	# Initialising list 'test_accuracy' with '0' that will later contain the test data accuracy of for K = i+1, where 'i' is the index value of the list 'test_accuracy'
# Loop over K values to get the train and test accuracy for different 'K' values
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors = k)																# Creating the classifier object for different 'K' values 
    knn.fit(X_train, y_train) 																		# Performing training
    # Compute traning and test data accuracy 
    train_accuracy[i] = knn.score(X_train, y_train) 
    test_accuracy[i] = knn.score(X_test, y_test) 
# Generating plot containing accuracy of training and test data for different 'K' values 
print("\n\t\tRESULT FOR K-NEAREST NEIGHBOR(KNN) CLASSIFIER FOR K VALUES RANGING FROM 1 TO 10")
fig = plt.figure()
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
plt.legend() 
plt.xlabel('K - Values') 
plt.ylabel('Accuracy') 
plt.savefig('KNN_Accuracies.png')
plt.close()
print("\nThe maximum accuracy for the test data is when 'K' =", test_accuracy.index(max(test_accuracy))+1, "and the accuracy(in %) is:", max(test_accuracy)*100)
knn = KNeighborsClassifier(n_neighbors = test_accuracy.index(max(test_accuracy))+1)											# Creating the classifier object for the K-value that gives maximum accuracy
knn.fit(X_train, y_train)																		# Performing training
y_pred = knn.predict(X_test)																		# Predicting values for the test data 'X_test'
print("The following is given for 'K' =", test_accuracy.index(max(test_accuracy))+1, ":")
print ("Confusion Matrix for KNN: \n", confusion_matrix(y_test, y_pred))												# Printing confusion matrix for KNN Classifier for the K-value that gives maximum accuracy
print("\n", classification_report(y_test, y_pred, zero_division = 0))													# Printing precision, recall, and f-score for the K-value that gives maximum accuracy


# Implementation of Naive Bayes
gnb = GaussianNB()																			# Creating the classifier object
gnb.fit(X_train, y_train)																		# Performing training
y_pred = gnb.predict(X_test)																		# Predicting values for the test data 'X_test'
print("\n\t\tRESULT FOR NAIVE BAYES CLASSIFIER")
print("\nGaussian Naive Bayes model accuracy(in %):", accuracy_score(y_test, y_pred)*100)										# Accuracy of the naive bayes
print ("Confusion Matrix for Naive Bayes: \n", confusion_matrix(y_test, y_pred))											# Printing confusion matrix
print("\n", classification_report(y_test, y_pred, zero_division = 0))													# Printing precision, recall, and f-score


# Implementation of Support Vector Machine
clf = SVC(kernel='linear') 																		# Creating the classifier object
clf.fit(X_train, y_train)																		# Performing training
y_pred = clf.predict(X_test)																		# Predicting values for the test data 'X_test'
print("\n\t\tRESULT FOR SUPPORT VECTOR MACHINE")
print("\nSupport Vector Machine accuracy(in %):", accuracy_score(y_test, y_pred)*100)											# Accuracy of the support vaector machine
print ("Confusion Matrix for Support Vector Machine: \n", confusion_matrix(y_test, y_pred))										# Printing confusion matrix
print("\n", classification_report(y_test, y_pred, zero_division = 0))													# Printing precision, recall, and f-score


# Implementation of Logistic Regression
classifier = LogisticRegression(random_state = 0, max_iter=1500) 													# Creating the classifier object
classifier.fit(X_train, y_train)																	# Performing training
y_pred = classifier.predict(X_test)																	# Predicting values for the test data 'X_test'
print("\n\t\tRESULT FOR LOGISTIC REGRESSION")
print("\nAccuracy for Logistic Regression(in %):", accuracy_score(y_test,y_pred)*100)											# Accuracy of the logistic regression
print ("Confusion Matrix for Logistic Regression: \n", confusion_matrix(y_test, y_pred))										# Printing confusion matrix
print("\n", classification_report(y_test, y_pred, zero_division = 0))													# Printing precision, recall, and f-score


# Implementation of cross validation for all the above trained classifiers
kfold =KFold(n_splits=10)
algos = ["Support Vector Machine", "K Nearest Neighbor", "Naive Bayes", "Decision Tree Gini", "Decision Tree Entropy"]
clfs = [svm.SVC(kernel = "linear"), KNeighborsClassifier(n_neighbors = 7), naive_bayes.GaussianNB(), tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5), tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)]
cv_results = []
for classifiers in clfs:
	cv_score = cross_val_score(classifiers, X, Y, cv = kfold, scoring = "accuracy")
	cv_results.append(cv_score.mean())
cv_mean = pd.DataFrame(cv_results,index=algos)
cv_mean.columns = ["Accuracy"]
cv_mean.sort_values(by = "Accuracy", ascending = False)
print("\nAccuracies for different classifiers using Cross Validation:\n\n", cv_mean, sep = "")
