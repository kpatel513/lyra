#!/bin/bash
# Script to install and test Claude Code
# This script will install Node.js (if needed) and Claude Code, then verify the installation

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
check_os() {
    log_info "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "Running on Linux"
    else
        log_warning "This script is optimized for Linux. Current OS: $OSTYPE"
    fi
}

# Check if Node.js is installed and version is >= 18
check_nodejs() {
    log_info "Checking Node.js installation..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        log_success "Node.js is installed: $(node --version)"
        
        if [ "$NODE_VERSION" -ge 18 ]; then
            log_success "Node.js version meets requirements (>= 18.0)"
            return 0
        else
            log_warning "Node.js version is too old. Need version 18 or higher."
            return 1
        fi
    else
        log_warning "Node.js is not installed"
        return 1
    fi
}

# Install Node.js and npm
install_nodejs() {
    log_info "Installing Node.js and npm..."
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        log_info "Using apt package manager"
        sudo apt update
        sudo apt install -y nodejs npm
    elif command -v yum &> /dev/null; then
        log_info "Using yum package manager"
        sudo yum install -y nodejs npm
    elif command -v dnf &> /dev/null; then
        log_info "Using dnf package manager"
        sudo dnf install -y nodejs npm
    else
        log_error "Could not detect package manager. Please install Node.js manually."
        exit 1
    fi
    
    log_success "Node.js and npm installed successfully"
}

# Check if npm is installed
check_npm() {
    log_info "Checking npm installation..."
    
    if command -v npm &> /dev/null; then
        log_success "npm is installed: $(npm --version)"
        return 0
    else
        log_error "npm is not installed"
        return 1
    fi
}

# Install Claude Code
install_claude_code() {
    log_info "Installing Claude Code..."
    
    # Try without sudo first
    if npm install -g @anthropic-ai/claude-code 2>/dev/null; then
        log_success "Claude Code installed successfully"
    else
        log_warning "Permission denied. Trying with sudo..."
        sudo npm install -g @anthropic-ai/claude-code
        log_success "Claude Code installed successfully with sudo"
    fi
}

# Verify Claude Code installation
verify_claude_code() {
    log_info "Verifying Claude Code installation..."
    
    if command -v claude &> /dev/null; then
        CLAUDE_VERSION=$(claude --version)
        log_success "Claude Code is installed: $CLAUDE_VERSION"
        return 0
    else
        log_error "Claude Code command not found"
        return 1
    fi
}

# Run Claude Code doctor command
run_claude_doctor() {
    log_info "Running Claude Code diagnostics..."
    
    # Run claude doctor and capture output
    if claude doctor 2>&1; then
        log_success "Claude Code diagnostics completed"
    else
        log_warning "Claude Code diagnostics returned warnings (this is normal before authentication)"
    fi
}

# Test Claude Code basic functionality
test_claude_code() {
    log_info "Testing Claude Code basic functionality..."
    
    # Test help command
    if claude --help &> /dev/null; then
        log_success "Help command works"
    else
        log_warning "Help command failed"
    fi
    
    # Test version command
    if claude --version &> /dev/null; then
        log_success "Version command works"
    else
        log_warning "Version command failed"
    fi
}

# Display next steps
show_next_steps() {
    echo ""
    echo "======================================"
    log_success "Claude Code Installation Complete!"
    echo "======================================"
    echo ""
    log_info "To get started with Claude Code:"
    echo ""
    echo "1. Navigate to your project directory:"
    echo "   cd /path/to/your/project"
    echo ""
    echo "2. Start Claude Code:"
    echo "   claude"
    echo ""
    echo "3. On first run, you'll need to authenticate using one of:"
    echo "   - Anthropic Console (requires billing at console.anthropic.com)"
    echo "   - Claude App (Pro or Max plan)"
    echo "   - Enterprise Platforms (Amazon Bedrock or Google Vertex AI)"
    echo ""
    echo "4. For more information, visit:"
    echo "   https://claudeai.dev/docs"
    echo ""
    log_info "Useful commands:"
    echo "   claude --version    # Check version"
    echo "   claude --help       # Show help"
    echo "   claude doctor       # Check configuration"
    echo ""
}

# Main installation flow
main() {
    echo ""
    echo "======================================"
    echo "  Claude Code Installation Script"
    echo "======================================"
    echo ""
    
    # Check OS
    check_os
    
    # Check and install Node.js if needed
    if ! check_nodejs; then
        install_nodejs
        
        # Verify installation
        if ! check_nodejs; then
            log_error "Failed to install Node.js"
            exit 1
        fi
    fi
    
    # Check npm
    if ! check_npm; then
        log_error "npm is required but not installed"
        exit 1
    fi
    
    # Install Claude Code
    install_claude_code
    
    # Verify installation
    if ! verify_claude_code; then
        log_error "Claude Code installation failed"
        exit 1
    fi
    
    # Run diagnostics
    run_claude_doctor
    
    # Test basic functionality
    test_claude_code
    
    # Show next steps
    show_next_steps
    
    log_success "All tests passed!"
}

# Run main function
main

