import subprocess
import sys
from termcolor import cprint

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

def install_requirements():
    """Install missing Python packages."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        cprint("[+] Missing packages installed successfully", "green")
    except subprocess.CalledProcessError as e:
        cprint(f"[-] Failed to install packages: {e}", "red")
        sys.exit(1)

def remove_duplicates(df):
    """Prompt the user to remove duplicate rows."""
    if df.duplicated().sum() == 0:
        cprint("[+] No duplicate data found!", "green")
        return
    if input("Do you want to remove duplicate data? (y/n): ").lower() == 'y':
        df.drop_duplicates(inplace=True)
        cprint("[+] Duplicate data removed!", "green")
    else:
        cprint("[+] Duplicate data not removed.", "blue")

def inspect_data(df):
    """Print a preview, information, and description of the DataFrame."""
    cprint("[*] Preview of dataset:", "blue")
    cprint(df.head(), "blue")
    cprint("\n\n[*] Data info:", "blue")
    cprint(df.info(), "blue")
    cprint("\n\n[*] Data description:", "blue")
    cprint(df.describe(), "blue")