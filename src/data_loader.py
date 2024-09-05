import pandas as pd
from termcolor import cprint
import sys

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

def save_dataframe(df, filename=None):
    """Prompt the user to save the DataFrame to a CSV file."""
    if filename is None:
        filename = input("Enter the filename to save the DataFrame: ")
    try:
        df.to_csv(filename, index=False)
        cprint("[+] DataFrame saved successfully!", "green")
    except Exception as e:
        cprint(f"[-] An error occurred while saving the DataFrame: {e}", "red")
