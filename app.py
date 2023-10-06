import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the Titanic train and test data
@st.cache_data
def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data.copy(), test_data.copy()

# Preprocess the data
def preprocess_data(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    if 'Cabin' in data.columns:
        data.drop('Cabin', axis=1, inplace=True)
    return data

# Train the model and generate predictions based on selected algorithm
def train_and_predict(train_data, algorithm):
    x = train_data[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = train_data['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    if algorithm == 'Logistic Regression':
        model = LogisticRegression(max_iter=500)
    elif algorithm == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    else:
        st.write("Please select a valid algorithm.")
        return None, None

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return y_test, predictions

# Display classification report
def display_metrics(y_test, predictions):
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    st.write(pd.DataFrame(cm, columns=['Predicted No', 'Predicted Yes'], index=['Actual No', 'Actual Yes']))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

    st.write("Classification Report:")

    # Get classification report as a string
    class_report = classification_report(y_test, predictions, target_names=['No', 'Yes'], output_dict=True)

    # Beautify the classification report
    class_report_str = ""
    for key in class_report.keys():
        if key in ['No', 'Yes']:
            class_report_str += f"\n**{key}**:\n"
            for metric, value in class_report[key].items():
                class_report_str += f"  - {metric}: {value:.2f}\n"
    
    st.markdown(class_report_str)

# Main Streamlit app
st.title('Titanic - Machine Learning from Disaster')
st.write("This app allows you to preprocess Titanic data, select a machine learning algorithm, and view metrics.")

# Load data
train_data, test_data = load_data()

# Preprocess train and test data
preprocessed_train_data = preprocess_data(train_data)
preprocessed_test_data = preprocess_data(test_data)

# Select machine learning algorithm
algorithm = st.selectbox("Select Machine Learning Algorithm", ['Logistic Regression', 'Random Forest'])

# Train the model and generate predictions
if st.button("Train and Predict"):
    y_test, predictions = train_and_predict(preprocessed_train_data, algorithm)

    # Display metrics
    if y_test is not None and predictions is not None:
        display_metrics(y_test, predictions)
    
