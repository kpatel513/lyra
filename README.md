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
   # or download the lyra directory to your desired location
   ```

2. **Make the scripts executable**
   ```bash
   cd lyra
   chmod +x lyra-summarize lyra-profile
   ```

3. **Add Lyra to your PATH (optional)**
   
   For global access from anywhere in your terminal:
   
   **For Zsh (macOS default):**
   ```bash
   echo 'export PATH="$PATH:/path/to/lyra"' >> ~/.zshrc
   source ~/.zshrc
   ```
   
   **For Bash:**
   ```bash
   echo 'export PATH="$PATH:/path/to/lyra"' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   Replace `/path/to/lyra` with the actual path to your lyra directory.

### Verify Installation

Test that Lyra is working correctly:

```bash
# If added to PATH:
lyra-summarize
lyra-profile

# Or use the full paths:
/path/to/lyra/lyra-summarize
/path/to/lyra/lyra-profile
```

You should see usage messages:
```
Usage: lyra-summarize REPO_PATH
Usage: lyra-profile REPO_PATH
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

### Performance Profiling Analysis (lyra-profile)

Analyze training pipelines for performance bottlenecks and profiling opportunities:

```bash
lyra-profile /path/to/your/ml/repository
```

**Example:**
```bash
lyra-profile ~/Work/my-pytorch-project
```

This generates a profiling analysis report covering:
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
- Ensure the scripts are executable: `chmod +x lyra-summarize lyra-profile`
- Check that the path is correct when added to PATH
- Try using the full path to the scripts

**Claude Code errors:**
- Ensure Claude Code is properly installed and authenticated
- Check that you're in a directory where Claude Code can run
