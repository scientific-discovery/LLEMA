#!/usr/bin/env python3
"""
LLEMA Setup Script
Run this script to set up the LLEMA environment for anonymous users
"""

import os
import sys
import subprocess
import shutil

def main():
    print("🚀 Setting up LLEMA for anonymous use...")
    
    # Check if we're in the right directory
    if not os.path.exists('src/agent/main.py'):
        print("❌ Error: Please run this script from the LLEMA root directory")
        sys.exit(1)
    
    # Create .env file from example
    if not os.path.exists('.env'):
        if os.path.exists('env.example'):
            shutil.copy('env.example', '.env')
            print("✅ Created .env file from env.example")
            print("⚠️  Please edit .env file and add your API keys")
        else:
            print("❌ Error: env.example file not found")
            sys.exit(1)
    else:
        print("✅ .env file already exists")
    
    # Check if conda is available
    try:
        subprocess.run(['conda', '--version'], check=True, capture_output=True)
        print("✅ Conda is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Conda not found. Please install Anaconda or Miniconda")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
    
    # Check if the mat_sci environment exists
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if 'mat_sci' in result.stdout:
            print("✅ mat_sci conda environment found")
        else:
            print("⚠️  mat_sci conda environment not found")
            print("   Please create it with: conda env create -f environment.yml")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Cannot check conda environments")
    
    # Make scripts executable
    scripts = ['src/run_all_tasks.sh', 'src/run_task.py']
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"✅ Made {script} executable")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Activate the conda environment: conda activate mat_sci")
    print("3. Run a single task: python src/run_task.py photovoltaic_absorbers")
    print("4. Run all tasks: ./src/run_all_tasks.sh")

if __name__ == "__main__":
    main()
