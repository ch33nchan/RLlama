import os
from typing import Any, Dict, Optional, Union

import numpy as np


class Logger:
    """
    Logger for tracking metrics during reinforcement learning experiments.
    
    Attributes:
        name: Name of the logger
        log_dir: Directory to save logs
        format: Format of the logs (e.g., "csv", "json")
        print_freq: Frequency of printing metrics
        metrics: Dictionary of metrics
        step: Current step
    """
    
    def __init__(
        self,
        name: str = "rllama",
        log_dir: Optional[str] = None,
        format: str = "csv",
        print_freq: int = 1000,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a logger.
        
        Args:
            name: Name of the logger
            log_dir: Directory to save logs
            format: Format of the logs (e.g., "csv", "json")
            print_freq: Frequency of printing metrics
            use_wandb: Whether to use Weights & Biases
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            wandb_config: Weights & Biases configuration
        """
        self.name = name
        self.log_dir = log_dir or f"logs/{name}"
        self.format = format
        self.print_freq = print_freq
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Initialize metrics
        self.metrics = {}
        self.step = 0
        
        # Initialize Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or name,
                    entity=wandb_entity,
                    config=wandb_config,
                    name=name,
                    dir=self.log_dir,
                )
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases not installed. Please install with `pip install wandb`.")
                self.use_wandb = False
                
        # Create output files
        if self.format == "csv":
            self.csv_file = open(os.path.join(self.log_dir, f"{name}.csv"), "w")
            self.csv_header_written = False
        elif self.format == "json":
            self.json_file = open(os.path.join(self.log_dir, f"{name}.json"), "w")
            self.json_file.write("[\n")
            self.json_file.flush()
            
    def record(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """
        Record a metric.
        
        Args:
            key: Name of the metric
            value: Value of the metric
            step: Step at which the metric was recorded
        """
        if step is None:
            step = self.step
            
        # Convert value to float or int if possible
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.item()
            
        # Store metric
        if key not in self.metrics:
            self.metrics[key] = []
            
        self.metrics[key].append((step, value))
        
        # Log to Weights & Biases
        if self.use_wandb:
            self.wandb.log({key: value}, step=step)
            
    def info(self, message: str) -> None:
        """
        Log an informational message.
        
        Args:
            message: Message to log
        """
        print(f"[{self.name}] {message}")
        
        # Write to log file
        with open(os.path.join(self.log_dir, f"{self.name}.log"), "a") as f:
            f.write(f"{message}\n")
            
    def warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
        """
        print(f"[{self.name}] WARNING: {message}")
        
        # Write to log file
        with open(os.path.join(self.log_dir, f"{self.name}.log"), "a") as f:
            f.write(f"WARNING: {message}\n")
            
    def error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
        """
        print(f"[{self.name}] ERROR: {message}")
        
        # Write to log file
        with open(os.path.join(self.log_dir, f"{self.name}.log"), "a") as f:
            f.write(f"ERROR: {message}\n")
            
    def step_print(self) -> None:
        """Print metrics if print_freq steps have passed."""
        if self.step % self.print_freq == 0:
            self.print_metrics()
            
        self.step += 1
        
    def print_metrics(self) -> None:
        """Print the latest metrics."""
        message = f"Step {self.step}: "
        
        for key, values in self.metrics.items():
            if values:
                latest_value = values[-1][1]
                message += f"{key}={latest_value:.4f} "
                
        self.info(message)
        
    def flush(self) -> None:
        """Flush metrics to disk."""
        if self.format == "csv":
            # Write header if needed
            if not self.csv_header_written and self.metrics:
                header = "step," + ",".join(self.metrics.keys())
                self.csv_file.write(header + "\n")
                self.csv_header_written = True
                
            # Write latest metrics
            latest_step = max([values[-1][0] for values in self.metrics.values()]) if self.metrics else 0
            row = f"{latest_step},"
            
            for key in self.metrics.keys():
                values = self.metrics[key]
                if values and values[-1][0] == latest_step:
                    row += f"{values[-1][1]},"
                else:
                    row += ","
                    
            self.csv_file.write(row.rstrip(",") + "\n")
            self.csv_file.flush()
            
        elif self.format == "json":
            # Write latest metrics as JSON
            latest_step = max([values[-1][0] for values in self.metrics.values()]) if self.metrics else 0
            json_obj = {"step": latest_step}
            
            for key in self.metrics.keys():
                values = self.metrics[key]
                if values and values[-1][0] == latest_step:
                    json_obj[key] = values[-1][1]
                    
            import json
            self.json_file.write(json.dumps(json_obj) + ",\n")
            self.json_file.flush()
            
    def close(self) -> None:
        """Close the logger and flush all metrics."""
        self.flush()
        
        if self.format == "csv":
            self.csv_file.close()
        elif self.format == "json":
            # Remove trailing comma and close JSON array
            self.json_file.seek(self.json_file.tell() - 2, os.SEEK_SET)
            self.json_file.write("\n]")
            self.json_file.close()
            
        # Close Weights & Biases
        if self.use_wandb:
            self.wandb.finish()