#!/bin/bash
# Setup script for Persian RAG System

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
print_section "Checking Python version"
if ! command_exists python3; then
    echo -e "${RED}Python 3 is required but not installed.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "Found Python ${GREEN}${PYTHON_VERSION}${NC}"

# Check and create virtual environment
print_section "Setting up Python virtual environment"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "Virtual environment ${GREEN}already exists${NC}."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_section "Upgrading pip"
pip install --upgrade pip

# Install Python dependencies
print_section "Installing Python dependencies"
pip install -r requirements.txt

# Check if Ollama is installed
print_section "Checking Ollama installation"
if ! command_exists ollama; then
    echo -e "${YELLOW}Ollama is not installed. Installing...${NC}"
    
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Add Ollama to PATH if not already in .bashrc or .zshrc
    if ! grep -q "export PATH=\"\$HOME/.ollama:\$PATH\"" ~/.bashrc; then
        echo 'export PATH="$HOME/.ollama:$PATH"' >> ~/.bashrc
        source ~/.bashrc
    fi
    
    echo -e "${GREEN}Ollama installed successfully.${NC}"
else
    echo -e "Ollama is ${GREEN}already installed${NC}."
fi

# Pull the required Ollama model
print_section "Checking Ollama models"
if ! ollama list | grep -q "qwen2.5"; then
    echo "Pulling qwen2.5 model..."
    ollama pull qwen2.5
else
    echo -e "qwen2.5 model ${GREEN}already exists${NC}."
fi

# Create necessary directories
print_section "Creating necessary directories"
mkdir -p processed_data raw_data

# Generate sample data if needed
if [ ! -f "processed_data/sample_data.csv" ]; then
    echo "Generating sample data..."
    python sample_data.py
else
    echo -e "Sample data ${GREEN}already exists${NC}."
fi

# Set execute permissions
chmod +x run_rag.py test_rag.py test_enhanced_rag.py test_system.py

# Print completion message
echo -e "\n${GREEN}✅ Setup completed successfully!${NC}"
echo -e "\nTo start the RAG system, run: ${YELLOW}source venv/bin/activate && ./run_rag.py${NC}"
echo -e "To run tests: ${YELLOW}source venv/bin/activate && python -m pytest${NC}"

# Deactivate virtual environment
deactivate
