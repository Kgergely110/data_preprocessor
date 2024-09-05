import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from src.data_loader import load_data, save_dataframe
from termcolor import cprint

class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up the environment before each test."""
        self.test_file = 'test_data.csv'
        self.test_data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Occupation': ['Engineer', 'Doctor', 'Artist']
        })
        # Create a temporary CSV file for testing
        self.test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_data(self):
        """Test loading a CSV file into a pandas DataFrame."""
        df = load_data(self.test_file)
        pd.testing.assert_frame_equal(df, self.test_data)
        cprint("[+] Test passed!", "green")
    
    def test_load_data_file_not_found(self):
        """Test behavior when the CSV file is not found."""
        with self.assertRaises(SystemExit):
            load_data("non_existent_file.csv")
        cprint("[-] Test passed!", "green")
    
    def test_save_dataframe(self):
        """Test saving a pandas DataFrame to a CSV file."""
        output_file = 'output_test_data.csv'
        save_dataframe(self.test_data, output_file)
        
        # Check if file is created
        self.assertTrue(os.path.exists(output_file))
        
        # Load the saved file and compare with original DataFrame
        saved_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(saved_df, self.test_data)
        
        cprint("[+] Test passed!", "green")
        # Clean up
        os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
