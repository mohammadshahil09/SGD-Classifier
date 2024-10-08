# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data
2. Split Dataset into Training and Testing Sets
3. Train the Model Using Stochastic Gradient Descent (SGD)
4. Make Predictions and Evaluate Accuracy
5. Generate Confusion Matrix


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SHAIK MOHAMMAD SHAHIL
RegisterNumber:212223240044  
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Iris dataset
iris = load_iris()

#Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

#Display the first few rows of the dataset
print(df.head())

#Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

#Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

#Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

#Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

#Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```


## Output:
![368415363-beeede28-eb80-4088-bb31-30d5294cb85d](https://github.com/user-attachments/assets/02b28051-7024-47ef-b269-29a00296d7bf)

![368415448-a58826f6-d3ab-4e40-ad63-d581bf93527d](https://github.com/user-attachments/assets/568ae8b9-6c6f-480e-8a36-74b8b9b3a3f6)

![368415536-59a53f3f-366a-4259-a79d-3f672048f48f](https://github.com/user-attachments/assets/bdbc6d41-e753-421b-9a79-0571b56f604d)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
