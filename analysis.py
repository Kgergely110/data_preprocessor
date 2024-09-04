"""_summary_

    Returns:
        _type_: _description_
    Author: Gergely Koncz
    Date: 2024-09-04
    Version: 1.0
"""
#Initial imports
import subprocess
import sys

# Command line argument check
if(len(sys.argv)<2):
    print("[-] Usage:\npython analysis.py <input_file_names_separated_with_space>")
    exit(1)


def download_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        cprint("[+] Missing packages installed successfully", "green")
    except subprocess.CalledProcessError as e:
        cprint("[-] Failed to install packages. Please check the requirements.txt file.", "red")
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        cprint("[-] An error occured while installing packages. Please check the requirements.txt file.", "red")
        print(f"Error: {e}")
        exit(1)

# Imports
i = 0
try:
    # For data loading, manipulation (preprocessing), analysis and visualization
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    # What is the best classifier for this dataset?
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression

    # To evaluate the model
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.metrics import f1_score

    # To calculate loss
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import log_loss
        
    # To load pre-trained models
    import joblib
    
    # Just something extra :)
    from termcolor import cprint
    
    cprint("[+] Packages imported successfully!", "green")
except:
    i += 1
    if(i<3):
        cprint("[-] Package requirements unsatisfied. Downloading necessary packages...", "red")
        download_requirements()
    else:
        cprint("[-] Imports could not be resolved. Please check your Python enviroment!", "red")
        exit(1)
        
# Loading the dataset
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        cprint("[-] Input file not found. Make sure it is in the same directory, as the .py file!", "red")
        exit(1)
    except Exception as e:
        print("An error occured: ", e)
        exit(1)
        
def inspect_data(df):
    cprint("[*] Preview of dataset:", "blue")
    cprint(df.head(), "blue")
    cprint("\n\n\n[*] Data info:", "blue")
    cprint(df.info(), "blue")
    cprint("\n\n\n[*] Data description:", "blue")
    cprint(df.describe(), "blue")

def missing_data(df):
    null_count = df.isnull().sum()
    if(null_count.sum() == 0):
        cprint("[+] No missing data found!", "green")
    else:
        cprint(f"[-] {null_count.sum()} missing values found (dataset size: {len(df)})!", "red")
        choice = 0
        cprint("[?] How would you like to handle missing data?\n", "yellow")
        while(choice not in range(1,8)):
            cprint("[1] Drop rows with missing values", "yellow")
            cprint("[2] Drop columns with missing values", "yellow")
            cprint("[3] Regression imputation", "yellow")
            cprint("[4] Mean imputation", "yellow")
            cprint("[5] Median imputation", "yellow")
            cprint("[6] Mode imputation", "yellow")
            cprint("[7] Custom imputation", "yellow")
            try:
                choice = int(input("Method number: "))
            except:
                choice = 0
            if(choice not in range(1,8)):
                cprint("\n\n[-] Invalid method number. Please try again!\n", "red")
        if choice == 1:
            drop_rows(df)
        elif choice == 2:
            drop_cols(df)
        else:
            cprint("[-] Method not yet implemented :(", "red")
            
def drop_rows(df):
    df.dropna(axis=0, inplace=True)
    cprint(f"[+] Rows with missing values have been dropped! {len(df)} records remaining.", "green")
    
def drop_cols(df):
    df.dropna(axis=1, inplace=True)
    cprint(f"[+] Columns with missing values have been dropped!", "green")
    
def regression_imputation(df):
    columns = pd.DataFrame(df).columns()
    for col in columns:
        if df[col].isnull().sum() > 0:
            cprint(f"[*] Missing values detected in column {col}", "blue")
            if df[col].dtype == 'object':
                cprint("[?] Cannot perform regression imputation on object-type column. How do you want to resolve this?\n", "yellow")
                choice = 0
                while (choice not in [1, 2]):
                    cprint("[1] Choose other method to fill missing data", "yellow")
                    cprint("[2] Ordinal encode column", "yellow")
                    try:
                        choice = int(input("Method number: "))
                    except:
                        choice = 0
                    if(choice not in range(1,3)):
                        cprint("\n\n[-] Invalid method number. Please try again!\n", "red")
                if choice == 1:
                    missing_data(df)
                else:
                    pass
                    # Ordinal encoding
            else:
                reg_model = LinearRegression()
                train_X = df[col]
                
    
    
    
    
def duplicate_data(df):
    answer = "q"
    while answer not in ["y", "Y", "n", "N"]:
        cprint("[?] Do you want to remove duplicate data (y/n)?", "yellow")
        answer = input("Choice: ")
        if answer not in  ["y", "Y", "n", "N"]:
            cprint("[-] Invlaid input. Please try again!", "red")
        
    if answer in ["y", "Y"]:
        pd.DataFrame(df).drop_duplicates()
        cprint(f"[+] Duplicate data removed. Remaining records: {len(df)}")
    else:
        cprint("[+] Duplicate data has NOT been removed.")
        
def main():
    files = sys.argv[1:]
    dataframes = []
    for file in files:
        df = load_data(file)
        inspect_data(df)
        missing_data(df)
        dataframes.append(df)    
    
main()