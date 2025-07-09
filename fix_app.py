#!/usr/bin/env python3

# This script will fix the missing if/elif structure in rllama_app.py

import re

# Path to your app file
app_file_path = "RLlama/rllama_app.py"

# Read the current content
with open(app_file_path, 'r') as file:
    content = file.read()

# Look for the problematic elif statement
if "elif page == \"ENVIRONMENTS\":" in content:
    # Insert a simple page variable declaration and if statement structure before it
    pattern = r"(.*)(elif page == \"ENVIRONMENTS\":)"
    replacement = r"\1# Page selection\npage = sidebar()\n\n# Main content based on page selection\nif page == \"ALGORITHMS\":\n    # Algorithm content here\n    pass\n\n\2"
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write the fixed content back
    with open(app_file_path, 'w') as file:
        file.write(fixed_content)
    
    print("✅ Fixed the missing if statement before elif in rllama_app.py")
else:
    print("❌ Couldn't find the problematic section. The file may have changed.")