import matplotlib.pyplot as plt
from termcolor import cprint
import seaborn as sns



def plot_menu(df):
    """Display a menu to the user to plot various types of graphs."""
    
    choices = [
        "Histogram",
        "Boxplot",
        "Scatter plot",
        "Correlation heatmap",
        "Back"
    ]
    
    while True:
        cprint("\n[*] Plot Menu:", "yellow")
        for i, choice in enumerate(choices, 1):
            cprint(f"[{i}] {choice}", "yellow")
        try:
            choice = int(input("Select an option: "))
            if choice == 1:
                plot_histogram(df)
            elif choice == 2:
                plot_boxplot(df)
            elif choice == 3:
                plot_scatter(df)
            elif choice == 4:
                plot_heatmap(df)
            elif choice == 5:
                cprint("[*] Returning to main menu...", "green")
                return
            else:
                cprint("[-] Invalid choice. Please try again!", "red")
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")
            
def plot_histogram(df):
    """Plot a histogram for the slected columns in the DataFrame."""
    while True:
        cprint("[?] Select columns to plot the histogram (space, comma and semicolon are delimiters, colon and dash define inclusive range)", "yellow")
        columns = df.columns
        for i, col in enumerate(columns, 1):
            cprint(f"[{i}] {col}", "yellow")
        try:
            choices = input("Enter column numbers: ").split()
            choices = [choice.strip() for choice in choices]
            choices = [int(choice) for choice in choices]
            if len(choices) == 0:
                cprint("[-] No columns selected. Please try again!", "red")
                continue
        except ValueError:
            cprint("[-] Invalid input. Please enter a number.", "red")
            continue
        if any(not choice.isdigit() for choice in choices or any(int(choice) < 1 or int(choice) > len(columns) for choice in choices)):
            cprint("[-] Invalid input. Please try again!", "red")
            continue
        
        for choice in choices:
            if '-' in choice:
                start, end = map(int, choice.split('-'))
                choices += list(range(start, end + 1))
            elif ':' in choice:
                start, end = map(int, choice.split(':'))
                choices += list(range(start, end + 1))
        break
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            plt.figure()
            sns.histplot(df[col].dropna())
            plt.title(f"Histogram of {col}")
            plt.show()
            return
            
def plot_boxplot(df):
    """Plot a boxplot for each column in the DataFrame."""
    columns = df.columns
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            plt.figure()
            sns.boxplot(x=col, data=df)
            plt.title(f"Boxplot of {col}")
            plt.show()
            return
        
def plot_scatter(df):
    """Plot a scatter plot for each pair of columns in the DataFrame."""
    columns = df.columns
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j and df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                plt.figure()
                sns.scatterplot(x=col1, y=col2, data=df)
                plt.title(f"Scatter plot of {col1} vs {col2}")
                plt.show()
                return
            
def plot_heatmap(df):
    """Plot a correlation heatmap for the DataFrame."""
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
    return

