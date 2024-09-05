from termcolor import cprint

def index_column(df):
    """Add or remove an index column from the DataFrame."""
    columns = df.columns
    index = [col for col in columns if ('index' in col.lower() or "id" in col.lower() or "key" in col.lower() or "idx" in col.lower()) and df[col].nunique() == len(df)]
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