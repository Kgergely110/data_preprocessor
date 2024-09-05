from .data_loader import load_data, save_dataframe
from .missing_data import handle_missing_data
from .categorical_data import handle_non_ordinal_column
from .model_menu import model_menu
from .plot_menu import plot_menu
from .column_operations import index_column, remove_column
from .utils import inspect_data, remove_duplicates
from .menu import menu
from .main import main


__version__ = "1.0"