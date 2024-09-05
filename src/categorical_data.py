import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from termcolor import cprint

def handle_non_ordinal_column(df, col):
    """Handle non-numeric columns when numeric imputation is attempted."""
    cprint(f"[!] Cannot perform numeric imputation on non-numeric column '{col}'.", "red")
    cprint("[?] How do you want to resolve the imputation into object-type column?", "yellow")

    choices = [
        "Change imputation method",
        "Ordinal encode the column",
        "Remove records with missing values and skip the column"
        "Drop the column"
    ]
    
    for i, choice in enumerate(choices, 1):
        cprint(f"[{i}] {choice}", "yellow")
    
    while True:
        try:
            choice = int(input("Select option number: "))
            if choice == 1:
                return
            elif choice == 2:
                if df[col].nunique() > 2:
                    cprint(f"[!] Column '{col}' has more than 2 unique values, please order them.", "yellow")
                    values = df[col].dropna().unique()
                    order = {}
                    while len(values) > 0:
                        for i, value in enumerate(values, 1):
                            cprint(f"[{i}] {value}", "yellow")
                        try:
                            current_value = int(input("Enter the smallest or least frequent value: ")) - 1
                            order[values[current_value]] = len(order)
                            values = np.delete(values, current_value)
                        except ValueError:
                            cprint("[-] Invalid input. Please enter a number.", "red")
                            continue
                        except IndexError:
                            cprint("[-] Invalid choice. Please try again!", "red")
                            continue

                    df[col].replace(order, inplace=True)
                    cprint(f"[+] Column '{col}' has been ordinal encoded.", "green")
                else:
                    ordinal_encoder = OrdinalEncoder()
                    df[col] = ordinal_encoder.fit_transform(df[[col]]).astype(int)
                    cprint(f"[+] Column '{col}' has been ordinal encoded.", "green")
                return
            elif choice == 3:
                df.dropna(subset=[col], inplace=True)
                cprint(f"[+] Records with missing values in column '{col}' have been removed.", "green")
                return
            elif choice == 4:
                df.drop(columns=[col], inplace=True)
                cprint(f"[+] Column '{col}' has been dropped.", "green")
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")