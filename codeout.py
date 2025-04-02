# ./codeout.py 
# Run this script to iterate over all python files in the project and compile a single file with all the code snippets. 
# The output file will be saved in the same directory as this script. 
# At the end you have a single file to provide to GPT and friends. 

import os
import re


# Settings
PROJECT_DIR = "."  # Change this if running from another folder
OUTPUT_FILE = "codeout_combined.py"
EXCLUDE_DIRS = {'__pycache__', '.git', 'venv', 'env', '.mypy_cache', 'build', 'dist', 'reports'}

def collect_python_files(base_dir):
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return sorted(python_files)

def extract_code(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Failed to read {file_path}: {e}\n"

def build_combined_file(file_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as out:
        for path in file_list:
            rel_path = os.path.relpath(path, PROJECT_DIR)
            out.write(f"\n\n# === FILE: {rel_path} ===\n\n")
            code = extract_code(path)
            out.write(code)

if __name__ == "__main__":
    print(f"[Info] Collecting Python files in '{PROJECT_DIR}'...")
    all_py_files = collect_python_files(PROJECT_DIR)
    print(f"[Info] {len(all_py_files)} files found.")

    print(f"[Info] Writing combined file to '{OUTPUT_FILE}'...")
    build_combined_file(all_py_files, OUTPUT_FILE)
    print("[Success] Combined code output generated.")
