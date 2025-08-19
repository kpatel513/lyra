import re
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProfileStep:
    """Represents a profiled step in the training pipeline."""
    name: str
    component_type: str  # LightningModule, Callback, Strategy, etc.
    total_time: float
    function_calls: int
    top_bottlenecks: List[Tuple[str, float, float]]  # (function, cumtime, percall)


@dataclass
class BottleneckAnalysis:
    """Analysis results for potential bottlenecks."""
    critical_steps: List[ProfileStep]
    time_distribution: Dict[str, float]
    recommendations: List[str]


class ProfilerAnalyzer:
    """Analyzes FIT profiler reports to identify bottlenecks in training pipelines."""
    
    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.profile_steps: List[ProfileStep] = []
        
    def parse_profiler_report(self) -> List[ProfileStep]:
        """Parse the profiler report and extract training pipeline steps."""
        steps = []
        current_step = None
        in_data_section = False
        
        with open(self.report_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                
                # Look for profile section headers
                if line.startswith("Profile stats for:"):
                    if current_step:
                        steps.append(current_step)
                    
                    # Extract component info
                    match = re.match(r'Profile stats for: \[(\w+)\](.*)', line)
                    if match:
                        component_type = match.group(1)
                        name = match.group(2).strip()
                        current_step = {
                            'name': name,
                            'component_type': component_type,
                            'bottlenecks': []
                        }
                        in_data_section = False
                
                # Extract timing information
                elif "function calls" in line and "seconds" in line and current_step:
                    match = re.search(r'(\d+) function calls.*in ([\d.]+) seconds', line)
                    if match:
                        current_step['function_calls'] = int(match.group(1))
                        current_step['total_time'] = float(match.group(2))
                
                # Start of data section
                elif line.startswith("ncalls") and current_step:
                    in_data_section = True
                
                # Extract individual function bottlenecks
                elif in_data_section and current_step and line and re.match(r'\s*\d+', line):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            cumtime = float(parts[3])
                            percall = float(parts[4]) if parts[4] != '0.000' else 0.0
                            function_name = ' '.join(parts[5:])
                            
                            # Only include significant bottlenecks
                            if cumtime > 0.1:  # More than 100ms
                                current_step['bottlenecks'].append((function_name, cumtime, percall))
                        except (ValueError, IndexError):
                            continue
                
                # Stop parsing function details after empty line
                elif in_data_section and not line:
                    in_data_section = False
        
        # Add the last step
        if current_step:
            steps.append(current_step)
        
        # Convert to ProfileStep objects
        self.profile_steps = []
        for step_data in steps:
            if 'total_time' in step_data and 'function_calls' in step_data:
                step = ProfileStep(
                    name=step_data['name'],
                    component_type=step_data['component_type'],
                    total_time=step_data['total_time'],
                    function_calls=step_data['function_calls'],
                    top_bottlenecks=sorted(step_data['bottlenecks'], 
                                         key=lambda x: x[1], reverse=True)[:10]
                )
                self.profile_steps.append(step)
        
        return self.profile_steps
    
    def identify_bottlenecks(self, min_time_threshold: float = 1.0) -> BottleneckAnalysis:
        """Identify potential bottlenecks in the training pipeline."""
        if not self.profile_steps:
            self.parse_profiler_report()
        
        # Find critical steps (>= threshold seconds)
        critical_steps = [step for step in self.profile_steps 
                         if step.total_time >= min_time_threshold]
        
        # Sort by total time
        critical_steps.sort(key=lambda x: x.total_time, reverse=True)
        
        # Calculate time distribution by component type
        time_by_component = {}
        total_time = sum(step.total_time for step in self.profile_steps)
        
        for step in self.profile_steps:
            comp_type = step.component_type
            time_by_component[comp_type] = time_by_component.get(comp_type, 0) + step.total_time
        
        # Generate recommendations
        recommendations = self._generate_recommendations(critical_steps, time_by_component)
        
        return BottleneckAnalysis(
            critical_steps=critical_steps,
            time_distribution=time_by_component,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, critical_steps: List[ProfileStep], 
                                time_by_component: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Data loading bottlenecks
        data_steps = [step for step in critical_steps 
                     if 'DataModule' in step.component_type or 'dataloader' in step.name.lower()]
        if data_steps:
            max_data_time = max(step.total_time for step in data_steps)
            if max_data_time > 60:  # More than 1 minute
                recommendations.append(
                    f"Data loading is taking {max_data_time:.1f}s. Consider: "
                    "1) Increase num_workers in DataLoader, "
                    "2) Use faster storage (SSD), "
                    "3) Pre-process and cache data, "
                    "4) Use memory mapping for large datasets"
                )
        
        # Training step bottlenecks
        training_steps = [step for step in critical_steps 
                         if 'training' in step.name.lower() or 'train' in step.name.lower()]
        for step in training_steps:
            if step.total_time > 30:
                recommendations.append(
                    f"Training step '{step.name}' takes {step.total_time:.1f}s. "
                    "Consider: 1) Reduce model complexity, 2) Optimize batch size, "
                    "3) Use mixed precision training, 4) Profile GPU utilization"
                )
        
        # I/O bottlenecks
        for step in critical_steps:
            io_functions = [func for func, time, _ in step.top_bottlenecks 
                           if any(keyword in func.lower() for keyword in 
                                ['read', 'write', 'load', 'save', 'acquire', 'lock'])]
            if io_functions:
                recommendations.append(
                    f"I/O bottleneck detected in '{step.name}': {io_functions[0]}. "
                    "Consider async I/O or reducing file operations"
                )
        
        # Memory-related bottlenecks
        for step in critical_steps:
            memory_functions = [func for func, time, _ in step.top_bottlenecks 
                              if any(keyword in func.lower() for keyword in 
                                   ['astype', 'copy', 'concatenate', 'concat'])]
            if memory_functions:
                recommendations.append(
                    f"Memory operation bottleneck in '{step.name}': {memory_functions[0]}. "
                    "Consider in-place operations or optimizing data types"
                )
        
        # Overall component analysis
        if 'LightningDataModule' in time_by_component:
            data_percentage = (time_by_component['LightningDataModule'] / 
                             sum(time_by_component.values())) * 100
            if data_percentage > 50:
                recommendations.append(
                    f"Data loading takes {data_percentage:.1f}% of total time. "
                    "This is the primary bottleneck - focus optimization efforts here"
                )
        
        return recommendations
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive bottleneck analysis report."""
        analysis = self.identify_bottlenecks()
        
        report_lines = [
            "🔍 Training Pipeline Bottleneck Analysis",
            "=" * 50,
            "",
            f"📊 Summary:",
            f"  • Total profiled steps: {len(self.profile_steps)}",
            f"  • Critical bottlenecks (≥1s): {len(analysis.critical_steps)}",
            f"  • Total time analyzed: {sum(step.total_time for step in self.profile_steps):.2f}s",
            "",
            "⏱️ Time Distribution by Component:",
        ]
        
        for component, time_spent in sorted(analysis.time_distribution.items(), 
                                          key=lambda x: x[1], reverse=True):
            percentage = (time_spent / sum(analysis.time_distribution.values())) * 100
            report_lines.append(f"  • {component}: {time_spent:.2f}s ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "🚨 Critical Bottlenecks:",
            ""
        ])
        
        for i, step in enumerate(analysis.critical_steps[:10], 1):
            report_lines.extend([
                f"{i}. {step.component_type}: {step.name}",
                f"   Time: {step.total_time:.2f}s | Calls: {step.function_calls:,}",
                "   Top bottleneck functions:"
            ])
            
            for func, cumtime, percall in step.top_bottlenecks[:5]:
                report_lines.append(f"     • {func}: {cumtime:.3f}s (avg: {percall:.4f}s)")
            
            report_lines.append("")
        
        report_lines.extend([
            "💡 Optimization Recommendations:",
            ""
        ])
        
        for i, rec in enumerate(analysis.recommendations, 1):
            report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report


def analyze_training_pipeline(report_path: str, output_file: Optional[str] = None) -> str:
    """
    Convenience function to analyze a FIT profiler report and identify bottlenecks.
    
    Args:
        report_path: Path to the FIT profiler report file
        output_file: Optional path to save the analysis report
        
    Returns:
        Formatted analysis report as string
    """
    analyzer = ProfilerAnalyzer(report_path)
    return analyzer.generate_report(output_file)


if __name__ == "__main__":
    # Example usage
    report_path = "/Users/msandora/Downloads/fit-profiler_report.txt"
    
    if os.path.exists(report_path):
        print("Analyzing training pipeline bottlenecks...")
        report = analyze_training_pipeline(report_path, "bottleneck_analysis.txt")
        print(report)
    else:
        print(f"Report file not found: {report_path}")