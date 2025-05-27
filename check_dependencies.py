#!/usr/bin/env python3
"""
Check if all required dependencies are installed.

This script checks if all the required Python packages are installed
and provides instructions for installing any missing dependencies.
"""

import sys
import subprocess
import pkg_resources

# Required packages and their minimum versions
REQUIRED_PACKAGES = {
    'requests': '2.31.0',
    'beautifulsoup4': '4.12.2',
    'pandas': '2.1.1',
    'numpy': '1.24.3',
    'hazm': '0.9.1',
    'langdetect': '1.0.9',
    'tqdm': '4.66.1',
    'pyarrow': '13.0.0',
    'fastparquet': '2023.4.0',
    'orjson': '3.9.10',
    'zstandard': '0.21.0',
    'clean-text': '0.6.0',
    'sentencepiece': '0.1.99',
    'protobuf': '4.25.0',
    'python-Levenshtein': '0.21.1',
    'pycld3': '0.22',
    'emoji': '2.8.0',
    'faiss-cpu': '1.7.4',
    'sentence-transformers': '2.2.2',
    'torch': '2.0.0',
    'transformers': '4.30.0',
    'langchain': '0.0.335',
    'langchain-community': '0.0.13',
    'langchain-core': '0.1.9',
    'langchain-text-splitters': '0.0.1',
}

def check_packages():
    """Check if all required packages are installed with correct versions."""
    missing_packages = []
    wrong_version = []
    
    for package, required_version in REQUIRED_PACKAGES.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version):
                wrong_version.append((package, installed_version, required_version))
        except pkg_resources.DistributionNotFound:
            missing_packages.append((package, required_version))
    
    return missing_packages, wrong_version

def install_package(package, version=None):
    """Install a package using pip."""
    package_spec = f"{package}>={version}" if version else package
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✅ Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_spec}")
        return False

def main():
    """Main function to check and install dependencies."""
    print("🔍 Checking dependencies...\n")
    
    missing_packages, wrong_version = check_packages()
    
    if not missing_packages and not wrong_version:
        print("✅ All required packages are installed with correct versions!")
        return True
    
    # Handle missing packages
    if missing_packages:
        print("\n❌ Missing packages:")
        for package, version in missing_packages:
            print(f"  - {package} (required: >={version})")
    
    # Handle wrong versions
    if wrong_version:
        print("\n⚠️  Wrong versions:")
        for package, installed, required in wrong_version:
            print(f"  - {package} (installed: {installed}, required: >={required})")
    
    # Ask user if they want to install missing packages
    if missing_packages or wrong_version:
        print("\nWould you like to install the missing or outdated packages? (y/n): ", end='')
        if input().strip().lower() == 'y':
            print("\nInstalling packages...\n")
            
            # Install missing packages first
            for package, version in missing_packages:
                install_package(package, version)
            
            # Upgrade packages with wrong versions
            for package, _, required in wrong_version:
                install_package(package, required)
            
            # Verify installation
            print("\nVerifying installation...")
            missing_packages, wrong_version = check_packages()
            
            if not missing_packages and not wrong_version:
                print("\n✅ All required packages are now installed with correct versions!")
                return True
            else:
                print("\n❌ Some packages could not be installed or updated.")
                return False
    
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nPlease install the missing or outdated packages manually using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
