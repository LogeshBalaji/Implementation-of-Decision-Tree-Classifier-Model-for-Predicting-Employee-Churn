# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Logesh B
RegisterNumber: 24900577 
*/
```
~~~
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
~~~
## Output:
# head()
![Screenshot 2024-11-23 233237](https://github.com/user-attachments/assets/7361c114-3b0c-4dbc-8a84-7f0e90e5d5d8)
# info()
![Screenshot 2024-11-23 233224](https://github.com/user-attachments/assets/85207983-0669-4da8-8042-5e0fefb52f61)
# NULL & COUNT
![Screenshot 2024-11-23 233210](https://github.com/user-attachments/assets/05d2422c-c079-4684-89c5-6620d21cffab)

![Screenshot 2024-11-23 233158](https://github.com/user-attachments/assets/12e98ed3-2564-4c15-bd3e-37c9ffefeeca)

![Screenshot 2024-11-23 233145](https://github.com/user-attachments/assets/08404c56-6024-4ef4-b5bf-32bfb6e2765f)

![Screenshot 2024-11-23 233124](https://github.com/user-attachments/assets/1994abe3-a349-471d-9ebb-76a72411449c)

# ACCURACY SCORE:
![Screenshot 2024-11-23 233103](https://github.com/user-attachments/assets/aaf1bc0b-6f8b-46d4-a26d-fb5a2aa73790)

# DECISION TREE CLASSIFIER MODEL:
![Screenshot 2024-11-23 233051](https://github.com/user-attachments/assets/8363eed1-90f0-4fa4-ba60-55bcdb2b086f)

![Screenshot 2024-11-23 233032](https://github.com/user-attachments/assets/696d07b6-c84b-4612-86a6-4ddda43e8185)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
