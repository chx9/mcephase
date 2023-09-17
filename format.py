import os
import subprocess
import glob

# Define the root directory to start searching for Python files.
root_dir = '.'


def format_python_code_in_directory(directory):
    python_files = glob.glob(os.path.join(
        directory, '**', '*.py'), recursive=True)

    for python_file in python_files:
        print(f"Formatting {python_file}")
        subprocess.run(f"autopep8 --in-place {python_file}", shell=True)


if __name__ == "__main__":
    format_python_code_in_directory(root_dir)
