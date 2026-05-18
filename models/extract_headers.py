import csv
import os

# Define the path to the CSV file relative to this script
# This assumes you save this script in the 'scripts' or 'models' folder.
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'symptoms_data.csv')


def print_csv_headers():
    try:
        # Open the CSV file
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)

            # The 'next' function grabs the very first row of the CSV
            headers = next(reader)

            print(f"Found {len(headers)} columns in the dataset:\n")
            print("-" * 40)

            # Print them in a numbered list for easy reading
            for i, header in enumerate(headers):
                print(f"{i + 1}. {header}")

            print("-" * 40)

            # Print them as a raw Python list so you can easily copy/paste them
            # if you need to update EXPECTED_FEATURES in ml_trainer.py
            print("\nRaw Python List (for easy copy-pasting):")
            print(headers)

    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {csv_path}")
        print("Make sure this script is in the 'scripts' or 'models' folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    print_csv_headers()