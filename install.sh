#!/bin/bash

# Lyra Installation Script
# Simplified installation for zsh users

set -e  # Exit on any error

echo "⚠️  DEPRECATED: This bash installer is kept for backward compatibility."
echo "✅ Recommended: install Lyra via Python instead:"
echo "   python -m venv .venv && source .venv/bin/activate && pip install -e \".[dev]\""
echo ""

echo "🚀 Installing Lyra..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LYRA_DIR="$SCRIPT_DIR"

echo "📂 Lyra directory: $LYRA_DIR"

# Ensure Python is available
PYTHON_BIN=""
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo "❌ Python not found!"
    echo "Please install Python 3.9+ and rerun."
    exit 1
fi

echo "✅ Using Python: $($PYTHON_BIN --version)"

# Create a local venv (recommended for this repo)
if [ ! -d "$LYRA_DIR/.venv" ]; then
    echo "🧪 Creating virtualenv at $LYRA_DIR/.venv..."
    "$PYTHON_BIN" -m venv "$LYRA_DIR/.venv"
fi

# Activate venv for installation + tests
echo "🪕 Activating venv and installing Lyra (editable)..."
source "$LYRA_DIR/.venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -e "$LYRA_DIR" >/dev/null

# Optional: dev extras
if [ "$1" == "--dev" ]; then
    echo "🧰 Installing dev dependencies..."
    python -m pip install -e "$LYRA_DIR[dev]" >/dev/null
fi

# Claude is optional now (needed only for lyra llm)
if command -v claude &> /dev/null; then
    echo "✅ Claude Code CLI found: $(claude --version 2>/dev/null || true)"
else
    echo "ℹ️  Claude Code CLI not found (only needed for: lyra llm ...)"
fi

# Optional PATH helper: add this repo's venv bin to PATH for zsh users.
if [ -n "$ZSH_VERSION" ] || [ -n "$BASH_VERSION" ]; then
    if echo "$PATH" | grep -q "$LYRA_DIR/.venv/bin"; then
        echo "✅ $LYRA_DIR/.venv/bin already on PATH for this shell"
    else
        echo "📝 (Optional) Add Lyra venv to PATH by adding this to ~/.zshrc:"
        echo "   export PATH=\"\$PATH:$LYRA_DIR/.venv/bin\""
    fi
fi

# Test installation (Python CLI)
echo "🧪 Testing installation..."
if lyra --version >/dev/null 2>&1; then
    echo "✅ lyra is working"
else
    echo "❌ lyra test failed"
    exit 1
fi

echo "🩺 Running: lyra check"
lyra check || true

echo ""
echo "🎉 Lyra installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate venv: source \"$LYRA_DIR/.venv/bin/activate\""
echo "   2. Run: lyra check"
echo "   3. Try: lyra summarize /path/to/repo"
echo ""
echo "📖 Usage examples:"
echo "   lyra summarize ~/my-ml-project"
echo "   lyra analyze ~/my-pytorch-training"
echo "   lyra profile ~/my-pytorch-training train.py --max-steps 100"
echo "   lyra llm analyze --repo ~/my-pytorch-training --profile-file lyra-profile.txt"
echo ""
echo "🔗 For more information, see: https://github.com/your-repo/lyra"