import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv('Dataset_spine.csv')
    data.rename(columns={
        "Col1": "pelvic_incidence",
        "Col2": "pelvic_tilt",
        "Col3": "lumbar_lordosis_angle",
        "Col4": "sacral_slope",
        "Col5": "pelvic_radius",
        "Col6": "degree_spondylolisthesis",
        "Col7": "pelvic_slope",
        "Col8": "direct_tilt",
        "Col9": "thoracic_slope",
        "Col10": "cervical_tilt",
        "Col11": "sacrum_angle",
        "Col12": "scoliosis_slope",
        "Class_att": "class"
    }, inplace=True)
    data = data.drop(columns=['Unnamed: 13'])
    data = data.drop_duplicates()
    data = data[data['degree_spondylolisthesis'] <= 400]  # Filter outliers
    X = data.drop(columns=['class'])
    y = data['class']
    return X, y

# Train and save the machine learning model
def train_and_save_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Train Logistic Regression
    logres = LogisticRegression()
    logres.fit(X_train, y_train)
    lr_accuracy = logres.score(X_test, y_test)



    # Save the best model (Logistic Regression) using pickle
    with open('ml_model.pkl', 'wb') as file:
        pickle.dump(logres, file)

    return  lr_accuracy

# Main function to load, preprocess, train, and save the model
def main():
    X, y = load_and_preprocess_data()
    lr_accuracy = train_and_save_model(X, y)
    print(f"Logistic Regression Accuracy: {lr_accuracy}")

if __name__ == "__main__":
    main()
