#!/usr/bin/env python3
"""
Advanced logging system for RLlama reward framework.
Provides comprehensive logging, monitoring, and analysis capabilities.
"""

import os
import json
import time
import csv
import pickle
from typing import Dict, Any, Optional, List, Union, TextIO
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict, deque
import numpy as np

class RewardLogger:
    """
    Advanced logger for reward values and components with multiple output formats.
    
    Features:
    - Multiple output formats (JSON, CSV, binary)
    - Real-time statistics tracking
    - Automatic log rotation
    - Thread-safe operations
    - Memory-efficient streaming
    - Performance analytics
    """
    
    def __init__(self, 
                 log_dir: str = "./reward_logs",
                 log_frequency: int = 100,
                 format: str = "json",
                 auto_save: bool = True,
                 max_memory_entries: int = 10000,
                 enable_rotation: bool = True,
                 rotation_size_mb: int = 100,
                 verbose: bool = False):
        """
        Initialize the reward logger.
        
        Args:
            log_dir: Directory where log files will be stored
            log_frequency: How often to write to disk (every N calls)
            format: Output format ("json", "csv", "binary", "all")
            auto_save: Whether to automatically save logs periodically
            max_memory_entries: Maximum entries to keep in memory
            enable_rotation: Whether to enable log file rotation
            rotation_size_mb: Size threshold for log rotation (MB)
            verbose: Whether to print logging info to console
        """
        self.log_dir = Path(log_dir)
        self.log_frequency = log_frequency
        self.format = format.lower()
        self.auto_save = auto_save
        self.max_memory_entries = max_memory_entries
        self.enable_rotation = enable_rotation
        self.rotation_size_mb = rotation_size_mb
        self.verbose = verbose
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.entries = deque(maxlen=max_memory_entries)
        self.call_count = 0
        self.session_start_time = time.time()
        
        # Statistics tracking
        self.stats = {
            'total_rewards': deque(maxlen=1000),
            'component_stats': defaultdict(lambda: deque(maxlen=1000)),
            'computation_times': deque(maxlen=1000),
            'error_count': 0,
            'last_save_time': time.time()
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # File handles for streaming
        self._file_handles = {}
        self._current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RewardLogger")
        
        # Initialize file handles based on format
        self._initialize_file_handles()
        
        if self.verbose:
            self.logger.info(f"RewardLogger initialized: {log_dir} (format: {format})")
    
    def _initialize_file_handles(self) -> None:
        """Initialize file handles for different output formats."""
        try:
            if self.format in ["json", "all"]:
                json_path = self.log_dir / f"rewards_{self._current_session_id}.json"
                self._file_handles['json'] = open(json_path, 'w')
                
            if self.format in ["csv", "all"]:
                csv_path = self.log_dir / f"rewards_{self._current_session_id}.csv"
                self._file_handles['csv'] = open(csv_path, 'w', newline='')
                self._csv_writer = csv.writer(self._file_handles['csv'])
                # Write header
                self._csv_writer.writerow([
                    'timestamp', 'step', 'total_reward', 'computation_time', 'components'
                ])
                
            if self.format in ["binary", "all"]:
                binary_path = self.log_dir / f"rewards_{self._current_session_id}.pkl"
                self._file_handles['binary'] = open(binary_path, 'wb')
                
        except Exception as e:
            self.logger.error(f"Error initializing file handles: {e}")
    
    def log_reward(self, 
                   total_reward: float, 
                   component_rewards: Dict[str, float],
                   step: Optional[int] = None,
                   context: Optional[Dict[str, Any]] = None,
                   computation_time: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a reward value with comprehensive information.
        
        Args:
            total_reward: The total reward value
            component_rewards: Dictionary of component-wise rewards
            step: Current step number (optional)
            context: Additional context to log (optional)
            computation_time: Time taken to compute reward (optional)
            metadata: Additional metadata (optional)
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Create log entry
            entry = {
                'timestamp': timestamp.isoformat(),
                'step': step if step is not None else self.call_count,
                'total_reward': float(total_reward),
                'component_rewards': {k: float(v) for k, v in component_rewards.items()},
                'computation_time': computation_time,
                'session_id': self._current_session_id,
                'call_count': self.call_count
            }
            
            # Add optional fields
            if context:
                entry['context'] = self._sanitize_context(context)
            if metadata:
                entry['metadata'] = metadata
                
            # Store in memory
            self.entries.append(entry)
            
            # Update statistics
            self._update_statistics(entry)
            
            # Write to files if streaming
            self._write_entry_to_files(entry)
            
            # Console output if verbose
            if self.verbose:
                self._log_to_console(entry)
            
            self.call_count += 1
            
            # Auto-save check
            if (self.auto_save and 
                self.call_count % self.log_frequency == 0):
                self._flush_buffers()
                
            # Rotation check
            if self.enable_rotation and self.call_count % 1000 == 0:
                self._check_rotation()
    
    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize context for logging (remove non-serializable objects)."""
        sanitized = {}
        
        for key, value in context.items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                # Convert to string representation
                sanitized[key] = str(value)
                
        return sanitized
    
    def _update_statistics(self, entry: Dict[str, Any]) -> None:
        """Update running statistics."""
        # Total reward stats
        self.stats['total_rewards'].append(entry['total_reward'])
        
        # Component stats
        for component, reward in entry['component_rewards'].items():
            self.stats['component_stats'][component].append(reward)
            
        # Computation time stats
        if entry['computation_time'] is not None:
            self.stats['computation_times'].append(entry['computation_time'])
    
    def _write_entry_to_files(self, entry: Dict[str, Any]) -> None:
        """Write entry to appropriate file formats."""
        try:
            # JSON format
            if 'json' in self._file_handles:
                json.dump(entry, self._file_handles['json'])
                self._file_handles['json'].write('\n')
                
            # CSV format
            if 'csv' in self._file_handles:
                self._csv_writer.writerow([
                    entry['timestamp'],
                    entry['step'],
                    entry['total_reward'],
                    entry.get('computation_time', ''),
                    json.dumps(entry['component_rewards'])
                ])
                
            # Binary format
            if 'binary' in self._file_handles:
                pickle.dump(entry, self._file_handles['binary'])
                
        except Exception as e:
            self.stats['error_count'] += 1
            self.logger.error(f"Error writing entry to files: {e}")
    
    def _log_to_console(self, entry: Dict[str, Any]) -> None:
        """Log entry information to console."""
        step = entry['step']
        total_reward = entry['total_reward']
        
        self.logger.info(f"Step {step}: Total Reward = {total_reward:.4f}")
        
        # Log component breakdown
        for component, reward in entry['component_rewards'].items():
            self.logger.info(f"  {component}: {reward:.4f}")
            
        # Log computation time if available
        if entry['computation_time'] is not None:
            self.logger.info(f"  Computation Time: {entry['computation_time']:.6f}s")
    
    def _flush_buffers(self) -> None:
        """Flush all file buffers."""
        try:
            for handle in self._file_handles.values():
                if hasattr(handle, 'flush'):
                    handle.flush()
                    
            self.stats['last_save_time'] = time.time()
            
            if self.verbose:
                self.logger.info(f"Flushed buffers at call {self.call_count}")
                
        except Exception as e:
            self.logger.error(f"Error flushing buffers: {e}")
    
    def _check_rotation(self) -> None:
        """Check if log files need rotation."""
        if not self.enable_rotation:
            return
            
        try:
            for format_name, handle in self._file_handles.items():
                if hasattr(handle, 'name'):
                    file_path = Path(handle.name)
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        
                        if size_mb > self.rotation_size_mb:
                            self._rotate_file(format_name)
                            
        except Exception as e:
            self.logger.error(f"Error checking rotation: {e}")
    
    def _rotate_file(self, format_name: str) -> None:
        """Rotate a log file."""
        try:
            # Close current file
            if format_name in self._file_handles:
                self._file_handles[format_name].close()
                
            # Create new session ID
            old_session_id = self._current_session_id
            self._current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Reinitialize file handle
            if format_name == "json":
                json_path = self.log_dir / f"rewards_{self._current_session_id}.json"
                self._file_handles['json'] = open(json_path, 'w')
                
            elif format_name == "csv":
                csv_path = self.log_dir / f"rewards_{self._current_session_id}.csv"
                self._file_handles['csv'] = open(csv_path, 'w', newline='')
                self._csv_writer = csv.writer(self._file_handles['csv'])
                self._csv_writer.writerow([
                    'timestamp', 'step', 'total_reward', 'computation_time', 'components'
                ])
                
            elif format_name == "binary":
                binary_path = self.log_dir / f"rewards_{self._current_session_id}.pkl"
                self._file_handles['binary'] = open(binary_path, 'wb')
                
            if self.verbose:
                self.logger.info(f"Rotated {format_name} log file: {old_session_id} -> {self._current_session_id}")
                
        except Exception as e:
            self.logger.error(f"Error rotating {format_name} file: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        with self._lock:
            total_rewards = list(self.stats['total_rewards'])
            computation_times = list(self.stats['computation_times'])
            
            stats = {
                'session_info': {
                    'session_id': self._current_session_id,
                    'start_time': datetime.fromtimestamp(self.session_start_time).isoformat(),
                    'duration_seconds': time.time() - self.session_start_time,
                    'total_calls': self.call_count,
                    'entries_in_memory': len(self.entries)
                },
                'reward_stats': {},
                'component_stats': {},
                'performance': {
                    'error_count': self.stats['error_count'],
                    'last_save_time': datetime.fromtimestamp(self.stats['last_save_time']).isoformat(),
                    'avg_calls_per_second': self.call_count / max(1, time.time() - self.session_start_time)
                }
            }
            
            # Total reward statistics
            if total_rewards:
                stats['reward_stats'] = {
                    'count': len(total_rewards),
                    'mean': np.mean(total_rewards),
                    'std': np.std(total_rewards),
                    'min': np.min(total_rewards),
                    'max': np.max(total_rewards),
                    'median': np.median(total_rewards),
                    'recent_mean': np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                }
                
            # Component statistics
            for component, values in self.stats['component_stats'].items():
                if values:
                    values_list = list(values)
                    stats['component_stats'][component] = {
                        'count': len(values_list),
                        'mean': np.mean(values_list),
                        'std': np.std(values_list),
                        'min': np.min(values_list),
                        'max': np.max(values_list),
                        'contribution': np.mean(values_list) / max(1e-8, np.mean(total_rewards)) if total_rewards else 0
                    }
                    
            # Performance statistics
            if computation_times:
                stats['performance'].update({
                    'avg_computation_time': np.mean(computation_times),
                    'max_computation_time': np.max(computation_times),
                    'min_computation_time': np.min(computation_times),
                    'total_computation_time': np.sum(computation_times)
                })
                
            return stats
    
    def save_logs(self, path: Optional[str] = None) -> str:
        """
        Save current logs to file.
        
        Args:
            path: Optional path to save logs to
            
        Returns:
            Path where logs were saved
        """
        with self._lock:
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str(self.log_dir / f"reward_logs_export_{timestamp}.json")
                
            # Prepare data for export
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'session_id': self._current_session_id,
                    'total_entries': len(self.entries),
                    'format_version': '1.0'
                },
                'statistics': self.get_statistics(),
                'entries': list(self.entries)
            }
            
            # Save to file
            try:
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
                if self.verbose:
                    self.logger.info(f"Logs exported to: {path}")
                    
                return path
                
            except Exception as e:
                self.logger.error(f"Error saving logs to {path}: {e}")
                raise
    
    def load_logs(self, path: str) -> Dict[str, Any]:
        """
        Load logs from a file.
        
        Args:
            path: Path to load logs from
            
        Returns:
            Loaded log data
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            if self.verbose:
                self.logger.info(f"Logs loaded from: {path}")
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading logs from {path}: {e}")
            raise
    
    def reset(self) -> None:
        """Reset the logger, clearing all stored values."""
        with self._lock:
            self.entries.clear()
            self.call_count = 0
            self.session_start_time = time.time()
            
            # Reset statistics
            self.stats['total_rewards'].clear()
            self.stats['component_stats'].clear()
            self.stats['computation_times'].clear()
            self.stats['error_count'] = 0
            self.stats['last_save_time'] = time.time()
            
            # Create new session
            old_session_id = self._current_session_id
            self._current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Close and reinitialize file handles
            self._close_file_handles()
            self._initialize_file_handles()
            
            if self.verbose:
                self.logger.info(f"Logger reset: {old_session_id} -> {self._current_session_id}")
    
    def _close_file_handles(self) -> None:
        """Close all file handles."""
        for handle in self._file_handles.values():
            try:
                if hasattr(handle, 'close'):
                    handle.close()
            except Exception as e:
                self.logger.error(f"Error closing file handle: {e}")
                
        self._file_handles.clear()
    
    def close(self) -> None:
        """Close the logger and all file handles."""
        with self._lock:
            self._flush_buffers()
            self._close_file_handles()
            
            if self.verbose:
                self.logger.info("RewardLogger closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
