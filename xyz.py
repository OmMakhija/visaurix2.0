import os
import ast
import subprocess
import sys

def get_project_files(directory):
    """
    Recursively gather all Python files in the specified directory, including subdirectories.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):  # Look for Python files
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def extract_imports(file_path):
    """
    Extract libraries from import statements in the given Python file.
    """
    libraries = set()  # Use a set to avoid duplicates
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    libraries.add(alias.name.split('.')[0])  # Only take the base package name
            elif isinstance(node, ast.ImportFrom):
                libraries.add(node.module.split('.')[0])  # For "from <module> import <something>"
    return libraries

def get_installed_libraries():
    """
    Get installed libraries and their versions using pip freeze.
    """
    installed_libraries = {}
    try:
        # Capture the output of 'pip freeze'
        freeze_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        
        # Parse the output into a dictionary of library names and versions
        for line in freeze_output.splitlines():
            if '==' in line:
                library, version = line.split('==')
                installed_libraries[library] = version
    except subprocess.CalledProcessError as e:
        print(f"Error getting installed libraries: {e}")
        sys.exit(1)

    return installed_libraries

def generate_requirements(project_directory, installed_libraries):
    """
    Generate a requirements.txt file for the project based on import statements.
    """
    # Get all Python files in the project
    python_files = get_project_files(project_directory)

    # Set to store all unique libraries
    all_libraries = set()

    # Extract libraries from each file
    for file_path in python_files:
        libraries = extract_imports(file_path)
        all_libraries.update(libraries)

    # List the libraries in requirements.txt format with versions
    requirements = []
    for library in sorted(all_libraries):
        if library in installed_libraries:
            requirements.append(f"{library}=={installed_libraries[library]}")
        else:
            print(f"Warning: {library} not found in pip freeze!")

    # Write to requirements.txt file
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for requirement in requirements:
            f.write(f"{requirement}\n")

    print("requirements.txt file generated successfully.")

if __name__ == "__main__":
    # Use the current working directory (since you're already in the project directory)
    project_directory = os.getcwd()

    # Get the installed libraries and their versions
    installed_libraries = get_installed_libraries()

    # Generate requirements.txt from the project files
    generate_requirements(project_directory, installed_libraries)
