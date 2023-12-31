# -*- coding: utf-8 -*-
"""CustomerChurn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17fyg7zMW0ZvMHlL5oMSeZDWwav66oP7A

Customer churn is a major problem and one of the most important concerns for large companies. Due to the direct effect on the revenues of the companies, especially in the telecom field, companies are seeking to develop means to predict potential customer churn. Therefore, finding factors that increase customer churn is important to take necessary actions to reduce this churn.

The model below is a multi-layer processing model that, when deployed, can take user input and return a categorical value for the user, showing whether there is a customer churn or not.

Importing libraries
"""

import pandas as pd
import os
import sklearn
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,f1_score


drive.mount('/content/drive')
df=pd.read_csv('/content/drive/My Drive/Colab Notebooks/CustomerChurn_dataset.csv')

"""Looking for most relevant features; viewing dataset"""

df.head()

df.info()

df.describe()

df.columns.tolist()

"""Dropping irrelevant columns"""

df.drop(columns=['customerID'], inplace = True)

df.columns.tolist()

"""Encoding data using the label encoder"""

from pandas.core.arrays import categorical

numericVariables = df.select_dtypes(include=['int64','float64'])
categoricalVariables = df.select_dtypes(include=['object'])

categoricalVariables = pd.DataFrame(categoricalVariables, columns =categoricalVariables.columns)
label_encoder = LabelEncoder()

for column in categoricalVariables.columns:
        categoricalVariables[column] = label_encoder.fit_transform(df[column])

"""Selecting most relevant features




"""

new_df = pd.concat([numericVariables, categoricalVariables], axis=1)
new_df.columns.tolist()

y = new_df['Churn']
X = new_df.drop('Churn', axis = 1)

X_df =  pd.DataFrame(X)

"""Using the Random Forest classifier to select most relevant features"""

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)
Xtrain.shape

# Create a tree-based model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# RFECV object
rfecv = RFECV(estimator=model, step=1, cv=3, scoring='accuracy')

rfecv.fit(Xtrain, Ytrain)

# selected features
selected_features = Xtrain.columns[rfecv.support_]

optimal_num_features = rfecv.n_features_
support_mask = rfecv.support_
selected_features = X.columns[support_mask]


selected_features

"""Pre-processing"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

scaled=sc.fit_transform(X)

scaled.tolist()

Xtrain.shape

"""Multi Layer Processing using Keras"""

import tensorflow as tf
from tensorflow import keras

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

!pip install --upgrade tensorflow

# Keras Functional API model
input_layer = Input(shape=(Xtrain.shape[1],))
hidden_layer_1 = Dense(32, activation='relu')(input_layer)
hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)
output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=50, batch_size=32, validation_data=(Xtest, Ytest))

"""Checking  Accuracy of the functional API model"""

_, accuracy = model.evaluate(Xtrain, Ytrain)
accuracy*100

loss, accuracy = model.evaluate(Xtest, Ytest)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.4f}')

"""Creating the model to allow the Gridsearch and Cross Validation implementation"""

def create_model(optimizer=Adam(learning_rate=0.0001), hidden_unit=32):
    input_layer = Input(shape=(Xtrain.shape[1],))
    hidden_layer_1 = Dense(hidden_unit, activation='relu')(input_layer)
    hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

"""Importing relevant Sklearn and Scikeras models and classifiers"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

"""Splitting the data  and applying random oversampling to the training data"""

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Split the data into train and test sets while preserving class distribution
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the RandomOverSampler
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

# Apply random oversampling to the training data
X_train_resampled, y_train_resampled = oversampler.fit_resample(Xtrain, Ytrain)

# Print the original and resampled class distribution
print("Original class distribution:", np.bincount(Ytrain))
print("Resampled class distribution:", np.bincount(y_train_resampled))

Ytrain.value_counts()

from sklearn.metrics import accuracy_score
from sklearn import metrics

num_classes=2
epochs=250
batch_size=64

"""Wrapping the Keras model using Keras Classifier and Using gridsearch to crossvalidate to find the best parameters for training"""

# Wrap the Keras model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

# Define the hyperparameter grid
param_grid = {
    'model__optimizer': ['adam','adadelta','rmsprop'],
    'model__hidden_unit': [32, 64, 128]
}

# Create GridSearchCV instance
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(Xtrain, Ytrain)

Viewing the best model to select parameters

best_model = grid_search.best_estimator_
best_model

from sklearn.metrics import classification_report

"""Checking Accuracy using AUC




"""

y_pred = best_model.predict(Xtest)
fpr_mlp, tpr_mlp, _ = metrics.roc_curve(Ytest, y_pred)
auc_mlp = round(metrics.roc_auc_score(Ytest, y_pred), 4)
print("AUC:",auc_mlp)
y_pred=np.round(best_model.predict(Xtest)).ravel()
print("\nCR by library method=\n",
          classification_report(Ytest, y_pred))

"""Saving the most suitable method for deployment"""

model.build_fn('adam',32).save("Model.h5")