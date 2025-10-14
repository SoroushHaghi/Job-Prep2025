import csv
import os

# مسیر فایل داده نسبت به ریشه پروژه تعریف می‌شود
DATA_FILE_PATH = os.path.join('data', 'accelerometer_data.csv')

def read_and_print_data_sample(file_path, num_lines=5):
    """
    چند خط اول از فایل CSV داده‌های سنسور را می‌خواند و چاپ می‌کند.
    """
    print(f"--- Reading sample data from: {file_path} ---")
    try:
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            header = next(csv_reader)
            print(f"Header: {', '.join(header)}")
            
            print("--- Data Samples: ---")
            for i, row in enumerate(csv_reader):
                if i < num_lines:
                    print(f"Row {i+1}: {', '.join(row)}")
                else:
                    break
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
        print("Please ensure you are running this script from the project's root directory (Job-Prep2025).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    read_and_print_data_sample(DATA_FILE_PATH)