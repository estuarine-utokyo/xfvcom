#!/bin/bash
# setup.sh - Installation script for xfvcom package

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if conda/mamba is installed
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    print_status "Found mamba installation"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    print_status "Found conda installation"
else
    print_error "Neither conda nor mamba is installed. Please install Miniforge or Anaconda first."
    echo "Visit: https://github.com/conda-forge/miniforge for installation instructions"
    exit 1
fi

# Parse command line arguments
ENV_NAME="xfvcom"
FORCE_RECREATE=false
INSTALL_DEV=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --force|-f)
            FORCE_RECREATE=true
            shift
            ;;
        --no-dev)
            INSTALL_DEV=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-name NAME    Name of the conda environment (default: xfvcom)"
            echo "  --force, -f        Remove existing environment and recreate"
            echo "  --no-dev           Skip development dependencies"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_status "Setting up xfvcom environment: $ENV_NAME"

# Check if environment already exists
if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
    if [ "$FORCE_RECREATE" = true ]; then
        print_warning "Removing existing environment: $ENV_NAME"
        $CONDA_CMD env remove -n $ENV_NAME -y
    else
        print_error "Environment '$ENV_NAME' already exists. Use --force to recreate."
        exit 1
    fi
fi

# Create conda environment from environment.yml
print_status "Creating conda environment from environment.yml..."
if [ "$INSTALL_DEV" = true ]; then
    $CONDA_CMD env create -f environment.yml -n $ENV_NAME
else
    # Create a temporary environment file without dev dependencies
    print_status "Creating environment without development dependencies..."
    grep -v -E "(pytest|mypy|black|isort|ipykernel|jupyterlab|types-|pandas-stubs)" environment.yml > /tmp/environment_no_dev.yml
    $CONDA_CMD env create -f /tmp/environment_no_dev.yml -n $ENV_NAME
    rm /tmp/environment_no_dev.yml
fi

# Initialize conda for the current shell if needed
eval "$($CONDA_CMD shell.bash hook)"

# Activate the environment
print_status "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Install xfvcom package in editable mode if not already installed via pip section
if ! pip show xfvcom &> /dev/null; then
    print_status "Installing xfvcom package in editable mode..."
    pip install -e .
fi

# Run basic import test
print_status "Testing xfvcom installation..."
if python -c "import xfvcom; print(f'xfvcom imported successfully from: {xfvcom.__file__}')" 2>/dev/null; then
    print_status "✓ xfvcom package imported successfully"
else
    print_error "Failed to import xfvcom package"
    exit 1
fi

# Test key dependencies
print_status "Verifying key dependencies..."
python -c "
import sys
try:
    import numpy
    import xarray
    import pandas
    import matplotlib
    import cartopy
    print('✓ Core dependencies verified')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
"

# Print environment information
print_status "Environment setup complete!"
echo ""
echo "========================================="
echo "  xfvcom Environment Setup Complete"
echo "========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""
echo "To activate this environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
if [ "$INSTALL_DEV" = true ]; then
    echo "Development tools installed:"
    echo "  - pytest (testing)"
    echo "  - mypy (type checking)"
    echo "  - black (code formatting)"
    echo "  - isort (import sorting)"
    echo "  - jupyterlab (notebook interface)"
    echo ""
    echo "Run tests with: pytest"
    echo "Format code with: black xfvcom/"
    echo "Check types with: mypy ."
fi
echo ""
echo "For more information, see README.md"