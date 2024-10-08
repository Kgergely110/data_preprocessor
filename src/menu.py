from termcolor import cprint
from utils import inspect_data
from data_loader import save_dataframe
from column_operations import index_column, remove_column
from plot_menu import plot_menu
from model_menu import model_menu
from categorical_data import handle_non_ordinal_column, choose_column

def menu(df, is_last=False):
    """Display a menu to the user to perform various operations on the DataFrame."""
    
    if is_last:
        choices = [
            "Add or remove index column",
            "Remove a column",
            "Deal with categorical data",
            "Train a classification model",
            "Save the dataframe",
            "Inspect data",
            "Plot menu",
            "Exit"
        ]
    else:
        choices = [
            "Add or remove index column",
            "Remove a column",
            "Deal with categorical data",
            "Train a classification model",
            "Save the dataframe",
            "Inspect data",
            "Plot menu",
            "Save and continue to next file",
            "Continue to next file without saving",
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
                col = choose_column(df)
                handle_non_ordinal_column(df, col)
            elif choice == 4:
                model_menu(df)
            elif choice == 5:
                save_dataframe(df)
            elif choice == 6:
                inspect_data(df)
            elif choice == 7:
                plot_menu(df)
            elif choice == 8:
                cprint("[*] Exiting...", "green")
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")