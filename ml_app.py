# import libraries

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# loading app

st.write("""
# Explore different ML model and datasets
Let see which is the best?  
""")

# put the data sets name in a box and then place it in sidebar

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Diabetes', 'Wine', 'Digits')
)

# next put the name of classifier in a box in sidebar

classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN','SVM', 'Random Forest')
)

# Now we define a function for uploading datsets

def get_dataset(dataset_name):
    data = None
    if dataset_name == "Diabetes":
        data = datasets.load_diabetes()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits()
    x = data.data
    y = data.target
    return x,y

    # now call the fucntion and assign it it equal to X, y

X, y = get_dataset(dataset_name)

# now print the shape of selected datasets

st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))


# next we add parameter of different classifiers in user input

def add_parameter_ui(classifier_name):
    params = dict() # Create an empty dictionary
    if classifier_name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = c # its the degree of correct classification
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['K'] = k  # its the namber of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth  # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators  # numbers of trees
    return params

# now call the fucntion and assign it  equal params

params = add_parameter_ui(classifier_name)
    

# making classifier based on classifier_name and params

def  get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf
    
    
# call the function and assign it equal to clf
clf = get_classifier(classifier_name, params)

# now split our datset into train and test data by 90/10 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)

# now train the classifier (model)

clf.fit(X_train, y_train)
y_pred = clf. predict(X_test)

# check accuracy score and print it on app

acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
# Now we draw our all feauters in two dimensional plot by using pca
pca = PCA(2)
X_projected = pca.fit_transform(X)

# now slice the data into o or 1 dimenssion
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig)