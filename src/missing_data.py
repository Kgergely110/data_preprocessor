from termcolor import cprint
import numpy as np
from categorical_data import handle_non_ordinal_column
from sklearn.linear_model import LinearRegression
import pandas as pd

def handle_missing_data(df):
    """Handle missing data in the DataFrame."""
    null_count = df.isnull().sum().sum()
    if null_count == 0:
        cprint("[+] No missing data found!", "green")
    else:
        cprint(f"\n[-] {null_count} missing values found!\n", "red")
        method_choice = get_imputation_method()
        impute_missing_data(df, method_choice)

def get_imputation_method(individual=False):
    """Prompt the user to select a method for handling missing data."""
    if individual:
        choices = [
            "Drop rows with missing values in this column",
            "Drop column",
            "Regression imputation",
            "Mean substitution",
            "Median substitution",
            "Mode substitution",
            "Custom value substitution"
        ]
    else:
        choices = [
            "Drop rows with missing values",
            "Drop columns with missing values",
            "Regression imputation",
            "Mean imputation",
            "Median imputation",
            "Mode imputation",
            "Custom value imputation",
            "Choose method for each column"
        ]
    for i, choice in enumerate(choices, 1):
        cprint(f"[{i}] {choice}", "yellow")
    
    while True:
        try:
            method = int(input("Select method number: "))
            if 1 <= method <= len(choices):
                return method
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")
            
def impute_missing_data(df:pd.DataFrame, method_choice, column=None):
    """Impute missing data in the DataFrame based on the selected method."""
    if method_choice == 1:
        df.dropna(axis=0, inplace=True)
        cprint("[+] Rows with missing values dropped!", "green")
    elif method_choice == 2:
        df.dropna(axis=1, inplace=True)
        cprint("[+] Columns with missing values dropped!", "green")
    else:
        if column is None:
            columns = df.columns   
            for col in columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype == 'object':
                        handle_non_ordinal_column(df, col)
                    if method_choice == 3:
                        regression_imputation(df, col)
                    elif method_choice == 4:
                        df[col].fillna(df[col].mean(), inplace=True)
                        cprint(f"[+] Filled missing values in '{col}' with mean!", "green")
                    elif method_choice == 5:
                        df[col].fillna(df[col].median(), inplace=True)
                        cprint(f"[+] Filled missing values in '{col}' with median!", "green")
                    elif method_choice == 6:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        cprint(f"[+] Filled missing values in '{col}' with mode!", "green")
                    elif method_choice == 7:
                        value = input(f"Provide a value to fill missing data in '{col}': ")
                        df[col].fillna(value, inplace=True)
                        cprint(f"[+] Filled missing values in '{col}' with custom value!", "green")
                    elif method_choice == 8:
                        individual_imputation(df)
        else:
            if df[column].dtype == 'object':
                handle_non_ordinal_column(df, column)
            if method_choice == 1:
                df[column].dropna(axis=0, inplace=True)
                cprint(f"[+] Rows with missing values in '{column}' dropped!", "green")
            elif method_choice == 2:
                df.drop(columns=[column], inplace=True)
                cprint(f"[+] Column '{column}' dropped!", "green")    
            elif method_choice == 3:
                regression_imputation(df, column)
            elif method_choice == 4:
                df[column].fillna(df[column].mean(), inplace=True)
                cprint(f"[+] Filled missing values in '{column}' with mean!", "green")
            elif method_choice == 5:
                df[column].fillna(df[column].median(), inplace=True)
                cprint(f"[+] Filled missing values in '{column}' with median!", "green")
            elif method_choice == 6:
                df[column].fillna(df[column].mode()[0], inplace=True)
                cprint(f"[+] Filled missing values in '{column}' with mode!", "green")
            elif method_choice == 7:
                value = input(f"Provide a value to fill missing data in '{column}': ")
                df[column].fillna(value, inplace=True)
                cprint(f"[+] Filled missing values in '{column}' with custom value!", "green")

def regression_imputation(df, col):
    """Perform regression imputation on the specified column."""
    not_null_df = df[df[col].notnull()]
    null_df = df[df[col].isnull()]
    
    X_train = not_null_df.drop(columns=[col])
    for column in X_train.columns:
        if X_train[column].dtype == 'object':
            handle_non_ordinal_column(X_train, column)
    y_train = not_null_df[col]
    
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    
    df.loc[df[col].isnull(), col] = reg_model.predict(null_df.drop(columns=[col]))
    cprint(f"[+] Filled missing values in '{col}' using regression imputation!", "green")
    
def individual_imputation(df):
    for col in df:
        if df[col].isnull().sum() > 0:
            cprint(f"\n[*] Current column: '{col}'.", "yellow")
            method_choice = get_imputation_method(individual=True)
            impute_missing_data(df, method_choice, column=col)