# lyra

Lyra is an AI Agent that automatically analyzes and optimizes ML training code before it runs on production clusters. It boosts model iteration speed, reduces GPU waste, and enforces training efficiency through profiling, smart code analysis, and detailed performance validation.

## Installation

### Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code/quickstart) installed and configured
- Bash shell (macOS/Linux)

### Install Lyra

1. **Clone or download the Lyra repository**
   ```bash
   git clone [your-repo-url] lyra
   cd lyra
   ```

2. **Run the installation script**
   ```bash
   ./install.sh
   ```

The install script will automatically:
- Check for Claude Code CLI
- Make scripts executable  
- Add Lyra to your PATH in `~/.zshrc`
- Test the installation
- Backup your existing `~/.zshrc`

### Verify Installation

Test that Lyra is working correctly:

```bash
# If added to PATH:
lyra-summarize
lyra-analyze

# Or use the full paths:
/path/to/lyra/lyra-summarize
/path/to/lyra/lyra-analyze
```

You should see usage messages:
```
Usage: lyra-summarize REPO_PATH
Usage: lyra-analyze REPO_PATH
```

## Usage

Lyra provides two main analysis commands:

### Repository Analysis (lyra-summarize)

Analyze any ML repository for mixed precision training and sharding implementations:

```bash
lyra-summarize /path/to/your/ml/repository
```

**Example:**
```bash
lyra-summarize ~/Work/my-pytorch-project
```

This generates a comprehensive report covering:
1. **Mixed Precision Training** - Detection of AMP, FP16/BF16 usage, GradScaler, etc.
2. **Sharding** - Analysis of distributed training, model parallelism, tensor sharding strategies

### Performance Analysis (lyra-analyze)

Analyze training pipelines for performance bottlenecks and profiling opportunities:

```bash
lyra-analyze /path/to/your/ml/repository
```

**Example:**
```bash
lyra-analyze ~/Work/my-pytorch-project
```

This generates a performance analysis report covering:
1. **Existing Profiling Setup** - Current profiler configurations and performance monitoring
2. **Training Pipeline Analysis** - Components that could benefit from profiling
3. **Performance Bottleneck Indicators** - Potential performance issues in the code
4. **Profiling Recommendations** - Specific strategies to optimize training performance

Both analyses are powered by Claude Code and provide detailed file locations, code snippets, and implementation details.

## Requirements

- Claude Code CLI must be installed and authenticated
- Target repositories should contain ML/AI training code (PyTorch, TensorFlow, JAX, etc.)

## Troubleshooting

**Command not found:**
- Ensure the scripts are executable: `chmod +x lyra-summarize lyra-analyze`
- Check that the path is correct when added to PATH
- Try using the full path to the scripts

**Claude Code errors:**
- Ensure Claude Code is properly installed and authenticated
- Check that you're in a directory where Claude Code can run
