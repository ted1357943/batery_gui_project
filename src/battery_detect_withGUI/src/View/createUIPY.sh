#!/bin/bash

# Set the directory path where you want to research fo .ui files
directory="$(dirname "$0")"

# Use the find command to search for all files in the specified directory and its subdirectories with a .ui extension
ui_files=$(find "$directory" -name "*.ui")

# Print the names of the .ui files found
echo "UI files found in $directory:"
echo "$ui_files"

# Convert each UI file to a Python file using pyuic5
echo "Start Convert .ui file to .py..."
for ui_file in $ui_files; do
    py_file="${ui_file%.ui}.py" # Replace .ui extension with .py extension
    pyuic5 -x "$ui_file" -o "$py_file"
    echo "$py_file"
done

# Print a message indicating the conversion is complete
echo "UI files converted to Python files in $directory"
