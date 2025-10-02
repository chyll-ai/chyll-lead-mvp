#!/usr/bin/env python3
"""
Simple Python server runner for Railway deployment
"""
import os
import sys
import subprocess

def main():
    # Install dependencies from root directory
    print("Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    
    # Change to the FastAPI directory
    os.chdir('services/fastapi')
    
    # Start the server
    print("Starting FastAPI server...")
    port = os.environ.get('PORT', '8000')
    subprocess.run([
        sys.executable, '-m', 'uvicorn', 
        'app-sirene-simple:app', 
        '--host', '0.0.0.0', 
        '--port', port
    ], check=True)

if __name__ == '__main__':
    main()
