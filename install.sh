#!/bin/bash

# Lyra Installation Script
# Simplified installation for zsh users

set -e  # Exit on any error

echo "🚀 Installing Lyra..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYRA_DIR="$SCRIPT_DIR"

echo "📂 Lyra directory: $LYRA_DIR"

# Check if Claude Code is installed
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code CLI not found!"
    echo "Please install Claude Code first: https://docs.anthropic.com/en/docs/claude-code/quickstart"
    exit 1
fi

echo "✅ Claude Code CLI found"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x "$LYRA_DIR/lyra-summarize"
chmod +x "$LYRA_DIR/lyra-analyze"
chmod +x "$LYRA_DIR/lyra-profile"

echo "✅ Scripts are now executable"

# Check if already in PATH
if echo "$PATH" | grep -q "$LYRA_DIR"; then
    echo "✅ Lyra is already in your PATH"
else
    # Add to zsh PATH
    echo "📝 Adding Lyra to your PATH in ~/.zshrc..."
    
    # Backup existing .zshrc if it exists
    if [ -f ~/.zshrc ]; then
        cp ~/.zshrc ~/.zshrc.backup.$(date +%Y%m%d_%H%M%S)
        echo "📋 Backed up existing ~/.zshrc"
    fi
    
    # Add PATH export to .zshrc
    echo "" >> ~/.zshrc
    echo "# Lyra CLI tools" >> ~/.zshrc
    echo "export PATH=\"\$PATH:$LYRA_DIR\"" >> ~/.zshrc
    
    echo "✅ Added Lyra to PATH in ~/.zshrc"
fi

# Test installation
echo "🧪 Testing installation..."

# Source the updated .zshrc for this session
export PATH="$PATH:$LYRA_DIR"

# Test commands
if "$LYRA_DIR/lyra-summarize" 2>&1 | grep -q "Usage:"; then
    echo "✅ lyra-summarize is working"
else
    echo "❌ lyra-summarize test failed"
    exit 1
fi

if "$LYRA_DIR/lyra-analyze" 2>&1 | grep -q "Usage:"; then
    echo "✅ lyra-analyze is working"
else
    echo "❌ lyra-analyze test failed"
    exit 1
fi

if "$LYRA_DIR/lyra-profile" 2>&1 | grep -q "Usage:"; then
    echo "✅ lyra-profile is working"
else
    echo "❌ lyra-profile test failed"
    exit 1
fi

echo ""
echo "🎉 Lyra installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Restart your terminal or run: source ~/.zshrc"
echo "   2. Test with: lyra-summarize"
echo "   3. Test with: lyra-analyze"
echo "   4. Test with: lyra-profile"
echo ""
echo "📖 Usage examples:"
echo "   lyra-summarize ~/my-ml-project"
echo "   lyra-analyze ~/my-pytorch-training"
echo "   lyra-profile ~/my-pytorch-training train.py"
echo ""
echo "🔗 For more information, see: https://github.com/your-repo/lyra"