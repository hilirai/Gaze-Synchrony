#!/bin/bash

# Gaze Synchronization Analysis Pipeline Setup Script
# Installs dependencies and sets up the environment for Linux systems

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "======================================================="
    echo "🔧  GAZE SYNCHRONIZATION PIPELINE SETUP  🔧"
    echo "======================================================="
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_python() {
    log_info "Checking Python version..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.10+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log_info "Found Python $PYTHON_VERSION"
    
    # Check if version is at least 3.10
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        log_success "Python version is compatible"
    else
        log_error "Python 3.10+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

check_gpu() {
    log_info "Checking for CUDA-capable GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        log_success "CUDA GPU detected: $GPU_INFO"
        return 0
    else
        log_warning "No CUDA GPU detected. Pipeline will use CPU (slower performance)."
        return 1
    fi
}

setup_virtual_environment() {
    log_info "Setting up virtual environment..."
    
    if [ -d ".venv" ]; then
        log_warning "Virtual environment already exists. Removing old one..."
        rm -rf .venv
    fi
    
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    log_success "Virtual environment created and activated"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Make sure we're in the virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        log_error "Virtual environment not activated. Please run: source .venv/bin/activate"
        exit 1
    fi
    
    # Install PyTorch with CUDA support if available
    if check_gpu; then
        log_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installing PyTorch CPU-only version..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    pip install -r requirements.txt
    
    log_success "Python dependencies installed"
}

setup_sam2() {
    log_info "Setting up SAM2 model..."
    
    # Check if SAM2 is already set up
    if [ -d "../sam2" ]; then
        log_warning "SAM2 directory already exists. Skipping SAM2 setup."
        log_info "If you need to reinstall SAM2, manually delete the '../sam2' directory"
        return 0
    fi
    
    # Clone and install SAM2
    cd ..
    if [ ! -d "segment-anything-2" ]; then
        log_info "Cloning SAM2 repository..."
        git clone https://github.com/facebookresearch/segment-anything-2.git
    fi
    
    cd segment-anything-2
    pip install -e .
    cd ../Gaze-Synchrony
    
    log_success "SAM2 model setup complete"
}

download_checkpoints() {
    log_info "Downloading SAM2 checkpoints..."
    
    if [ -f "./download_ckpts.sh" ]; then
        chmod +x ./download_ckpts.sh
        ./download_ckpts.sh
        log_success "SAM2 checkpoints downloaded"
    else
        log_warning "download_ckpts.sh not found. You may need to download checkpoints manually."
    fi
}

create_data_directories() {
    log_info "Creating data directories..."
    
    mkdir -p data/raw
    mkdir -p data/processed/gaze_csv
    mkdir -p data/processed/sync_csv  
    mkdir -p data/processed/figures
    mkdir -p docs
    
    log_success "Data directories created"
}

verify_installation() {
    log_info "Verifying installation..."
    
    # Test Python imports
    python3 -c "
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import flask
print('✓ All core dependencies imported successfully')
"
    
    # Test SAM2 import
    if python3 -c "from sam2.build_sam import build_sam2_video_predictor; print('✓ SAM2 imported successfully')" 2>/dev/null; then
        log_success "SAM2 is properly installed"
    else
        log_warning "SAM2 import failed. You may need to install it manually."
    fi
    
    log_success "Installation verification complete"
}

print_usage_instructions() {
    echo ""
    echo -e "${GREEN}=======================================================${NC}"
    echo -e "${GREEN}🎉  SETUP COMPLETE!  🎉${NC}"
    echo -e "${GREEN}=======================================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo -e "   ${YELLOW}source .venv/bin/activate${NC}"
    echo ""
    echo "2. Place your video files in:"
    echo -e "   ${YELLOW}data/raw/${NC}"
    echo ""
    echo "3. Run the analysis pipeline:"
    echo -e "   ${YELLOW}cd scripts${NC}"
    echo -e "   ${YELLOW}./run_gaze_analysis.sh player1.mp4 player2.mp4 cooperation${NC}"
    echo ""
    echo "4. View results in:"
    echo -e "   ${YELLOW}data/processed/${NC}"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

main() {
    print_banner
    
    # Check system requirements
    check_python
    
    # Setup environment
    setup_virtual_environment
    install_python_dependencies
    setup_sam2
    download_checkpoints
    create_data_directories
    
    # Verify everything works
    verify_installation
    
    # Show usage instructions
    print_usage_instructions
    
    log_success "Setup completed successfully!"
}

# Run main function
main "$@"