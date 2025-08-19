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
lyra-profile

# Or use the full paths:
/path/to/lyra/lyra-summarize
/path/to/lyra/lyra-analyze
/path/to/lyra/lyra-profile
```

You should see usage messages:
```
Usage: lyra-summarize REPO_PATH
Usage: lyra-analyze REPO_PATH
Usage: lyra-profile REPO_PATH
```

## Usage

Lyra provides three main commands for comprehensive ML training analysis:

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

### Safe Profiling (lyra-profile)

Generate profiler data by running training code in safe mode:

```bash
lyra-profile /path/to/your/ml/repository [TRAINING_SCRIPT]
```

**Examples:**
```bash
# Auto-detect training script
lyra-profile ~/Work/my-pytorch-project

# Profile specific training script
lyra-profile ~/Work/my-pytorch-project train.py

# Profile script in subdirectory
lyra-profile ~/Work/my-pytorch-project scripts/train_model.py
```

This safely profiles your training pipeline by:
1. **Environment Setup** - Creates isolated environment and installs dependencies automatically
2. **Safe Mode Operation** - Disables model saving, prevents data modification, limits to 100 steps
3. **Advanced Profiling** - Adds PyTorch Lightning AdvancedProfiler for detailed timing analysis
4. **Automated Execution** - Runs modified training code and generates profiler reports
5. **Clean Restoration** - Restores all temporarily modified files and cleans up environments

The generated profiler report can then be analyzed using `lyra-analyze` for bottleneck identification.

### Workflow Integration

For comprehensive analysis, use the commands in sequence:

```bash
# 1. Analyze repository structure and optimizations
lyra-summarize ~/my-project

# 2. Generate profiler data safely  
lyra-profile ~/my-project train.py

# 3. Analyze profiler output for bottlenecks
lyra-analyze ~/my-project
```

All analyses are powered by Claude Code and provide detailed file locations, code snippets, and implementation details.

## Requirements

- Claude Code CLI must be installed and authenticated
- Target repositories should contain ML/AI training code (PyTorch, TensorFlow, JAX, etc.)

## Troubleshooting

**Command not found:**
- Ensure the scripts are executable: `chmod +x lyra-summarize lyra-analyze lyra-profile`
- Check that the path is correct when added to PATH
- Try using the full path to the scripts

**Claude Code errors:**
- Ensure Claude Code is properly installed and authenticated
- Check that you're in a directory where Claude Code can run
