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

# Check if mamba is installed (preferred over conda for supercomputer compatibility)
if command -v mamba &> /dev/null; then
    MAMBA_CMD="mamba"
    print_status "Found mamba installation"
else
    print_error "Mamba is not installed. Please install Miniforge first."
    echo "Visit: https://github.com/conda-forge/miniforge for installation instructions"
    echo "Note: Use mamba instead of conda to avoid conflicts with Intel oneAPI Python."
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
if $MAMBA_CMD env list | grep -q "^$ENV_NAME "; then
    if [ "$FORCE_RECREATE" = true ]; then
        print_warning "Removing existing environment: $ENV_NAME"
        $MAMBA_CMD env remove -n $ENV_NAME -y
    else
        print_error "Environment '$ENV_NAME' already exists. Use --force to recreate."
        exit 1
    fi
fi

# Create environment from environment.yml using mamba
print_status "Creating mamba environment from environment.yml..."
if [ "$INSTALL_DEV" = true ]; then
    mamba env create -f environment.yml -n $ENV_NAME
else
    # Create a temporary environment file without dev dependencies
    print_status "Creating environment without development dependencies..."
    grep -v -E "(pytest|mypy|black|isort|ruff|ipykernel|jupyterlab|types-|pandas-stubs)" environment.yml > /tmp/environment_no_dev.yml
    mamba env create -f /tmp/environment_no_dev.yml -n $ENV_NAME
    rm /tmp/environment_no_dev.yml
fi

# Initialize conda for the current shell if needed
eval "$(conda shell.bash hook)"

# Activate the environment
print_status "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Ensure the environment resolved to Python 3.13
# if [[ $PYTHON_VERSION != 3.13.* ]]; then
#    print_error "Expected Python 3.13.x but found $PYTHON_VERSION."
#    print_error "Ensure python=3.13 is specified and recreate the environment."
#    exit 1
# fi

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
