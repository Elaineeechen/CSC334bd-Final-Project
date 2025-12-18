# CSC334bd Final Project

## Problem I'm trying to solve
I'm trying to predict whether a patient has diabetes based on diagnostic measurements from the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data). This Kaggle dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, consists of female patients at least 21 years old of Pima Indian heritage.

## How my solution is structured
My solution was organized into two main steps.  
The first step, data exploration and preprocessing, focused on understanding the dataset and preparing it for modeling. This included looking at the summary statistics of features, identifying missing values, and applying data imputation to estimate those missing values. I also checked for class imbalance, which was found to not be a major concern for this dataset.  
The second step involved developing machine learning models to predict diabetes. Initial models were first trained using default hyperparameters. I then tuned hyperparameters, combined with cross-validation, to improve accuracy. For the black-box model, I also attempted to improve its explainability by generating feature importances.

## Types of models my solution uses
I used K-Nearest Neighbors (KNN) and Multi-layer Perceptron Classifier (MLPClassifier). KNN is a distanced-based algorithm that classifies data points based on the majority class of their closest neighbors. MLPClassifier is a feedforward neural network model that learns patterns through multiple layers of interconnected neurons, trained with backpropagation. 

## A minimal reproducible example / tutorial
The `diabetes.ipynb` notebook contains the full implementation along with explanations. This section provides a minimal example using a small toy dataset to demonstrate the basic workflow of data preparation and model training. 

1. Run `ipython` in terminal
```bash
$ ipython
Python 3.10.4 (main, Mar 31 2022, 08:41:55) [GCC 7.5.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.3.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: 
```

2. Import required libraries
```python
In [1]: import pandas as pd
In [2]: from sklearn.model_selection import train_test_split
In [3]: from sklearn.neighbors import KNeighborsClassifier
In [4]: from sklearn.preprocessing import StandardScaler
In [5]: from sklearn.neural_network import MLPClassifier
```

3. Construct a small toy dataset
```python
In [6]: data = {
   ...:     "Pregnancies": [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
   ...:     "Glucose": [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
   ...:     "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 73, 70, 96],
   ...:     "SkinThickness": [35, 29, 21, 23, 35, 22, 32, 31, 45, 33],
   ...:     "Insulin": [212, 62, 271, 94, 168, 123, 88, 142, 543, 153],
   ...:     "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 36.02478],
   ...:     "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
   ...:     "Age": [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
   ...:     "Outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
   ...: }
In [7]: df = pd.DataFrame(data)
```

4. Split the data
```python
In [8]: y = df['Outcome']
In [9]: X = df.drop(['Outcome'], axis = 1)
In [10]: X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 0)
```

5. Scale features
```python
In [11]: scaler = StandardScaler()
In [12]: X_train_scaled = scaler.fit_transform(X_train)
In [13]: X_test_scaled = scaler.transform(X_test)
```

6. Train and evaluate models
```python 
In [14]: knn_init = KNeighborsClassifier()
In [15]: knn_init.fit(X_train_scaled, y_train)
Out[15]: KNeighborsClassifier()

In [16]: train_score = knn_init.score(X_train_scaled, y_train)
In [17]: test_score = knn_init.score(X_test_scaled, y_test)
In [18]: print(f"At default k = {5}, KNN training accuracy = {train_score} & test accuracy = {test_score}")
At default k = 5, KNN training accuracy = 0.75 & test accuracy = 0.5

In [19]: mlp_init = MLPClassifier(random_state = 42)
In [20]: mlp_init.fit(X_train_scaled, y_train)
/opt/miniconda3/lib/python3.12/site-packages/sklearn/neural_network/_multilayer_perceptron.py:781: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
Out[20]: MLPClassifier(random_state=42)

In [21]: print(f"At default settings, MLPClassifier training accuracy = {mlp_init.score(X_train_scaled, y_train)}")
At default settings, MLPClassifier training accuracy = 1.0
In [22]: print(f"Test accuracy = {mlp_init.score(X_test_scaled, y_test)}")
Test accuracy = 0.5
```

Note again that this small toy dataset is for a simple demonstration purpose only and is not intended for meaningful performance evaluation. The `diabetes.ipynb` notebook contains the full implementation, including data preprocessing, hyperparameter tuning, and feature importance discussion omitted from this section. 

## URLs of resources used
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data  
https://www.kaggle.com/code/leedonghyeok/multiple-imputation-mice  
https://www.kaggle.com/code/shrutimechlearn/step-by-step-diabetes-classification#Hyper-Parameter-optimization  
https://www.kaggle.com/code/harshwardhanfartale/explainable-ai-interpreting-ml-models
