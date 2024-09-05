from termcolor import cprint
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def model_menu(df):
    """
        Prompt the user to train a classification model on the current DataFrame.
    """
    
    choices = [
        "Train a Decision Tree Classifier",
        "Train a Random Forest Classifier",
        "Train a Linear Regression Model",
        "Back"
    ]
    
    while True:
        cprint("\n[*] Model Menu:", "yellow")
        for i, choice in enumerate(choices, 1):
            cprint(f"[{i}] {choice}", "yellow")
        try:
            choice = int(input("Select an option: "))
            if choice == 1:
                train_decision_tree(df)
            elif choice == 2:
                train_random_forest(df)
            elif choice == 3:
                train_linear_regression(df)
            elif choice == 4:
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")
            
def train_decision_tree(df):
    """
        Train a Decision Tree Classifier on the DataFrame.
    """
    
    target = select_target(df)
    features = select_features(df, target)
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cprint(classification_report(y_test, y_pred), "green")
    cprint(f"Accuracy: {model.score(X_test, y_test)}", "green")
    
def train_random_forest(df):
    """
        Train a Random Forest Classifier on the DataFrame.
    """
    
    target = select_target(df)
    features = select_features(df, target)
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cprint(classification_report(y_test, y_pred), "green")
    cprint(f"Accuracy: {model.score(X_test, y_test)}", "green")
    
def train_linear_regression(df):
    """
        Train a Linear Regression model on the DataFrame.
    """
    
    target = select_target(df)
    features = select_features(df, target)
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cprint(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}", "green")
    cprint(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}", "green")
    cprint(f"R2 Score: {r2_score(y_test, y_pred)}", "green")
    
def select_target(df):
    """
        Prompt the user to select the target column for the model.
    """
    
    cprint("\n[*] Select the target column for the model:", "blue")
    for i, col in enumerate(df.columns, 1):
        cprint(f"[{i}] {col}", "blue")
    choice = int(input("Enter the number corresponding to the target column: "))
    return df.columns[choice-1]

def select_features(df, target):
    """
        Prompt the user to select the feature columns for the model.
    """
    
    cprint("\n[*] Select the feature columns for the model (separate by commas):", "blue")
    for i, col in enumerate(df.columns, 1):
        cprint(f"[{i}] {col}", "blue")
    choices = input("Enter column numbers: ").replace(',', ' ').replace(';', ' ').split()
    choices = [choice.strip() for choice in choices]
    
    return [df.columns[int(choice)-1] for choice in choices if df.columns[int(choice)-1] != target] 