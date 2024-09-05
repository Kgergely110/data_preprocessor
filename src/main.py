"""
A data analysis and preprocessing script.

Author: Gergely Koncz
Date: 2024-09-04
Version: 1.1
"""

import sys
import warnings
from termcolor import cprint
warnings.filterwarnings("ignore")

from data_loader import load_data
from missing_data import handle_missing_data
from utils import safe_import, remove_duplicates, inspect_data
from menu import menu

def main():
    if len(sys.argv) < 2:
        print("[-] Usage:\npython main.py <input_file_names_separated_with_space>")
        sys.exit(1)

    safe_import()
    
    files = sys.argv[1:]
    for file in files:
        df = load_data(file)
        inspect_data(df)
        handle_missing_data(df)
        remove_duplicates(df)
        cprint("[*] Initial preprocessing complete!", "green")
        if  file == files[-1]:
            menu(df, is_last=True)
        else:
            menu(df)

if __name__ == "__main__":
    main()
