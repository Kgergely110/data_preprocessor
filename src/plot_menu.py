import matplotlib.pyplot as plt
from termcolor import cprint
import seaborn as sns


def select_columns(df):
    """Prompt the user to select columns and return the corresponding column names."""
    while True:
        cprint("[?] Select columns to plot (space, comma and semicolon are delimiters, colon and dash define inclusive range)", "yellow")
        columns = df.columns
        for i, col in enumerate(columns, 1):
            cprint(f"[{i}] {col}", "yellow")
        
        try:
            choices = input("Enter column numbers: ").replace(',', ' ').replace(';', ' ').split()
            choices = [choice.strip() for choice in choices]
            selected_columns = []
            
            for choice in choices:
                if '-' in choice or ':' in choice:
                    start, end = map(int, choice.replace('-', ':').split(':'))
                    selected_columns += list(columns[start-1:end])
                else:
                    selected_columns.append(columns[int(choice) - 1])
                    
            if not selected_columns:
                cprint("[-] No columns selected. Please try again!", "red")
                continue
            
            return selected_columns
        
        except (ValueError, IndexError):
            cprint("[-] Invalid input. Please enter valid column numbers.", "red")
            continue


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
            choice = int(input("Select an option: ").strip())
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
    """Plot a histogram for the selected columns in the DataFrame."""
    selected_columns = select_columns(df)
    
    for col in selected_columns:
        if df[col].dtype in ['int64', 'float64']:
            plt.figure()
            sns.histplot(df[col], kde=True)
            plt.title(f"Histogram of {col}")
            plt.show()


def plot_boxplot(df):
    """Plot a boxplot for the selected columns in the DataFrame."""
    selected_columns = select_columns(df)
    
    for col in selected_columns:
        if df[col].dtype in ['int64', 'float64']:
            plt.figure()
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()


def plot_scatter(df):
    """Plot a scatter plot for a pair of selected columns in the DataFrame."""
    selected_columns = select_columns(df)
    
    if len(selected_columns) >= 2:
        for i, col1 in enumerate(selected_columns):
            for col2 in selected_columns[i+1:]:
                if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                    plt.figure()
                    sns.scatterplot(x=col1, y=col2, data=df)
                    plt.title(f"Scatter plot of {col1} vs {col2}")
                    plt.show()
    else:
        cprint("[-] Please select at least two columns for scatter plot.", "red")


def plot_heatmap(df):
    """Plot a correlation heatmap for the DataFrame with only numeric columns."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty:
        cprint("[-] No numeric columns found for correlation heatmap.", "red")
        return
    elif len(numeric_df.columns) == 1:
        cprint("[-] There are less, then two numeric columns.", "red")
        return
    elif len(numeric_df.columns) != len(df.columns):
        cprint(F"[!] Warning: {len(df.columns)-len(numeric_df.columns)} non-numeric columns are ignored in correlation heatmap.", "yellow")

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


