"""
A data analysis and preprocessing script.

Author: Gergely Koncz
Date: 2024-09-04
Version: 1.1
"""
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score,
                             mean_absolute_error, mean_squared_error, log_loss)
import joblib
from termcolor import cprint
from sklearn.preprocessing import OrdinalEncoder

def install_requirements():
    """Install missing Python packages listed in requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        cprint("[+] Missing packages installed successfully", "green")
    except subprocess.CalledProcessError as e:
        cprint(f"[-] Failed to install packages: {e}", "red")
        sys.exit(1)
    except Exception as e:
        cprint(f"[-] An error occurred while installing packages: {e}", "red")
        sys.exit(1)

def safe_import():
    """Import required packages and install them if they are missing."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                                     roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score,
                                     mean_absolute_error, mean_squared_error, log_loss)
        import joblib
        from termcolor import cprint
        from sklearn.preprocessing import OrdinalEncoder
        cprint("[+] Packages imported successfully!", "green")
    except ImportError:
        cprint("[-] Package requirements unsatisfied. Downloading necessary packages...", "red")
        install_requirements()

def load_data(filename):
    """Load a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        cprint("[-] Input file not found. Ensure it is in the same directory as the script!", "red")
        sys.exit(1)
    except Exception as e:
        cprint(f"[-] An error occurred while loading data: {e}", "red")
        sys.exit(1)

def inspect_data(df):
    """Print a preview, information, and description of the DataFrame."""
    cprint("[*] Preview of dataset:", "blue")
    cprint(df.head(), "blue")
    cprint("\n\n[*] Data info:", "blue")
    cprint(df.info(), "blue")
    cprint("\n\n[*] Data description:", "blue")
    cprint(df.describe(), "blue")

def handle_missing_data(df):
    """Handle missing data in the DataFrame."""
    null_count = df.isnull().sum().sum()
    if null_count == 0:
        cprint("[+] No missing data found!", "green")
    else:
        cprint(f"[-] {null_count} missing values found!", "red")
        method_choice = get_imputation_method()
        impute_missing_data(df, method_choice)

def get_imputation_method():
    """Prompt the user to select a method for handling missing data."""
    choices = [
        "Drop rows with missing values",
        "Drop columns with missing values",
        "Regression imputation",
        "Mean imputation",
        "Median imputation",
        "Mode imputation",
        "Custom imputation"
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

def handle_non_ordinal_column(df, col):
    """Handle non-numeric columns when numeric imputation is attempted."""
    cprint(f"[!] Cannot perform numeric imputation on non-numeric column '{col}'.", "red")
    cprint("[?] You can either change the method, ordinal encode the values, or drop the column.", "yellow")

    choices = [
        "Change imputation method",
        "Ordinal encode the column",
        "Drop the column"
    ]
    for i, choice in enumerate(choices, 1):
        cprint(f"[{i}] {choice}", "yellow")
    
    while True:
        try:
            choice = int(input("Select option number: "))
            if choice == 1:
                handle_missing_data(df)
                return
            elif choice == 2:
                if df[col].nunique() > 2:
                    cprint(f"[!] Column '{col}' has more than 2 unique values, please order them.", "yellow")
                    values = df[col].unique().exclude('nan')
                    order = {}
                    while len(values) > 0:
                        for i, value in enumerate(values, 1):
                            cprint(f"[{i}] {value}", "yellow")
                        current_value = input("Enter the smallest or less frequent value: ")
                        try:
                            order[values[int(current_value) - 1]] = len(order)
                            values = np.delete(values, int(current_value) - 1)
                        except ValueError:
                            cprint("[-] Invalid input. Please enter a number.", "red")
                            continue
                        except IndexError:
                            cprint("[-] Invalid choice. Please try again!", "red")
                            continue
                        except Exception as e:
                            cprint(f"[-] An error occurred: {e}", "red")
                            continue
                        for key, value in order.items():
                            df[col].replace(key, value, inplace=True)
                    cprint(f"[+] Column '{col}' has been ordinal encoded.", "green")
                    
                else:
                    ordinal_encoder = OrdinalEncoder()
                    df[col] = ordinal_encoder.fit_transform(df[[col]]).astype(int)
                    cprint(f"[+] Column '{col}' has been ordinal encoded.", "green")
                    return
            elif choice == 3:
                df.drop(columns=[col], inplace=True)
                cprint(f"[+] Column '{col}' has been dropped.", "green")
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")

def impute_missing_data(df, method_choice):
    """Impute missing data in the DataFrame based on the selected method."""
    if method_choice == 1:
        df.dropna(axis=0, inplace=True)
        cprint("[+] Rows with missing values dropped!", "green")
    elif method_choice == 2:
        df.dropna(axis=1, inplace=True)
        cprint("[+] Columns with missing values dropped!", "green")
    else:
        columns = df.columns
        for col in columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    handle_non_ordinal_column(df, col)
                elif method_choice == 3:
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

def remove_duplicates(df):
    """Prompt the user to remove duplicate rows in the DataFrame."""
    if input("Do you want to remove duplicate data? (y/n): ").lower() == 'y':
        df.drop_duplicates(inplace=True)
        cprint("[+] Duplicate data removed!", "green")
    else:
        cprint("[+] Duplicate data not removed.", "blue")

def menu(df):
    """Display a menu to the user to perform various operations on the DataFrame."""
    choices = [
        "Add or remove index column",
        "Remove a column",
        "Train a classification model",
        "Save the dataframe",
        "Plot menu",
        "Exit"
    ]
    
    while True:
        cprint("\n[*] Menu:", "yellow")
        for i, choice in enumerate(choices, 1):
            cprint(f"[{i}] {choice}", "yellow")
        try:
            choice = int(input("Select an option: "))
            if choice == 1:
                index_column(df)
            elif choice == 2:
                remove_column(df)
            elif choice == 3:
                model_menu(df)
            elif choice == 4:
                save_dataframe(df)
            elif choice == 5:
                plot_menu(df)
            elif choice == 6:
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")
            
            
def index_column(df):
    """Add or remove an index column from the DataFrame."""
    columns = df.columns
    index = [col for col in columns if 'index' in col.lower() or "id" in col.lower() or "key" in col.lower() or "idx" in col.lower()]
    if len(index) == 0:
        cprint("[!] No index column found!", "blue")
        cprint("[+] Do you want to add an index column? (y/n)", "green")
        if input("Choice: ").lower() == 'y':
            df.insert(0, 'index', range(1, 1 + len(df)))
            cprint("[+] Index column added!", "green")
        else:
            cprint("[+] Index column not added.", "blue")
        return
    else:
        index_column = index[0]
        cprint(f"[!] Index column found: {index_column}", "blue")
        cprint(f"[+] Do you want to remove the index column? (y/n)", "green")
        if input().lower() == 'y':
            df.drop(columns=[index_column], inplace=True)
            cprint("[+] Index column removed!", "green")
        else:
            cprint("[+] Index column not removed.", "blue")
        
def remove_column(df):
    columns = df.columns
    cprint("[*] Columns in the dataset:", "blue")
    for i, col in enumerate(columns, 1):
        cprint(f"[{i}] {col}", "blue")
    while True:
        try:
            choice = input("Select column number to remove or type \'back\' to return to menu: ")
            if choice == 'back':
                menu(df)
                break
            choice = int(choice)
            if 1 <= choice <= len(columns):
                df.drop(columns=[columns[choice - 1]], inplace=True)
                cprint("[+] Column removed!", "green")
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")

def save_dataframe(df):
    """Prompt the user to save the DataFrame to a CSV file."""
    filename = input("Enter the filename to save the DataFrame: ")
    try:
        df.to_csv(filename, index=False)
        cprint("[+] DataFrame saved successfully!", "green")
    except Exception as e:
        cprint(f"[-] An error occurred while saving the DataFrame: {e}", "red")

def main():
    """Main function to orchestrate data loading, inspection, and preprocessing."""
    if len(sys.argv) < 2:
        print("[-] Usage:\npython analysis.py <input_file_names_separated_with_space>")
        sys.exit(1)

    safe_import()
    
    files = sys.argv[1:]
    for file in files:
        df = load_data(file)
        inspect_data(df)
        handle_missing_data(df)
        remove_duplicates(df)
        cprint("[*] Preprocessing complete!", "green")
        menu(df)

if __name__ == "__main__":
    main()