#!/bin/bash

# Set the directory path where you want to research fo .ui files
directory="$(dirname "$0")"


# Use the find command to search for all files in the specified directory and its subdirectories with a .ui extension
ui_files=$(find "$directory" -name "*.ui")

resource_qrc="resource_for_mirdc.qrc"

resource_py="${resource_qrc%.qrc}_rc.py"


# Print the names of the .ui files found
echo "Resource qrc file is $resource_qrc"

# Convert each UI file to a Python file using pyuic5
echo "Start Convert .qrc file to .py..."
pyrcc5 -o "$resource_py" "$resource_qrc"

# Print a message indicating the conversion is complete
echo "Converted Python files : $resource_py"

