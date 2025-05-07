"""
Run script for Advanced OCR System that fixes PyTorch-Streamlit compatibility issues

This script sets the necessary environment variables and runs the OCR application
with the file watcher disabled to prevent conflicts with PyTorch.
"""
import os
import sys

# Set environment variables to fix compatibility issues
os.environ["STREAMLIT_SERVER_WATCH_FILES"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"  # Helps with PyTorch loading issues

# Import streamlit after setting environment variables
import streamlit.web.bootstrap as bootstrap

# Run the app with file watcher disabled
if __name__ == "__main__":
    print("Starting Advanced OCR System with compatibility fixes...")
    print("Using environment variables to disable file watching.")
    
    # Run the application with special flags
    bootstrap.run(
        "streamlit_app.py",  # The main application file
        "",  # Command line arguments (empty)
        [],  # Script args
        flag_options={
            'server.fileWatcherType': 'none',  # Disable file watcher
            'browser.serverAddress': 'localhost',
            'server.headless': True,
            'server.runOnSave': False,
            'global.developmentMode': False
        }
    )
