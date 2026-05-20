# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the employee dataset using dummy variable encoding.

2.Split the dataset into training and testing data and initialize the Decision Tree classifier.

3.Train the Decision Tree model using the training dataset.

4.Predict test results, calculate accuracy, and visualize the Decision Tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: sribalakumaran.r
RegisterNumber:  212225220104
*//*



import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()



print("data.info():")
data.info()



print("isnull() and sum():")
data.isnull().sum()



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()



print("x.head():")
# Fixed: Added closing quote to complete the string literal "Work_accident"
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident"]]
x.head()



y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy



print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0]])  # Removed the last 2 values (1,2)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()

```

## Output:
<img width="881" height="291" alt="image" src="https://github.com/user-attachments/assets/f25fa8b7-225f-49fd-a1b1-a6599eca32e0" />

<img width="480" height="393" alt="image" src="https://github.com/user-attachments/assets/aee6863d-3e20-48a7-a58b-af8062d8a2ae" />

<img width="282" height="282" alt="image" src="https://github.com/user-attachments/assets/91e9b5f1-30f4-4690-971f-715216bde4e4" />

<img width="258" height="122" alt="image" src="https://github.com/user-attachments/assets/67344f22-94ba-4c86-933b-2a72383a1746" />

<img width="887" height="260" alt="image" src="https://github.com/user-attachments/assets/fd7ec711-3a95-4aa4-ac9d-c02ec5c55e7e" />

<img width="908" height="257" alt="image" src="https://github.com/user-attachments/assets/df71b942-3868-4166-8a0c-8fdf5bcf2e8c" />

<img width="197" height="67" alt="image" src="https://github.com/user-attachments/assets/389330ed-2c76-4cdb-ab66-f44466415775" />

<img width="907" height="682" alt="image" src="https://github.com/user-attachments/assets/54f26edd-6377-421a-a8dd-657a0538ac53" />








## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
