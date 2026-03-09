"""
Performance Benchmarker
Measures FPS, latency, memory, and power consumption
"""

import torch
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import json

from src.utils.logger import setup_logger


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: Device to benchmark on
        """
        self.device = device
        self.logger = setup_logger("Benchmarker")
        
    def benchmark_model(
        self,
        model: torch.nn.Module,
        input_size: tuple = (1, 3, 640, 640),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model performance
        
        Args:
            model: Model to benchmark
            input_size: Input tensor size  
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations
            
        Returns:
            Dict of performance metrics
        """
        self.logger.info("="*80)
        self.logger.info("Performance Benchmarking")
        self.logger.info("="*80)
        
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Warmup
        self.logger.info(f"Warming up ({warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        self.logger.info(f"Benchmarking ({num_iterations} iterations)...")
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                
                _ = model(dummy_input)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        metrics = {
            'fps': 1000 / np.mean(latencies),
            'latency_mean_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
            'latency_min_ms': np.min(latencies),
            'latency_max_ms': np.max(latencies),
            'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99),
        }
        
        # Memory usage
        if self.device == "cuda":
            metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1e6
            metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1e6
        
        # CPU memory
        process = psutil.Process()
        metrics['cpu_memory_mb'] = process.memory_info().rss / 1e6
        
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        metrics['model_size_mb'] = param_size / 1e6
        
        # Parameter count
        metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in formatted table"""
        self.logger.info("="*80)
        self.logger.info("Benchmark Results")
        self.logger.info("="*80)
        self.logger.info(f"FPS:                {metrics['fps']:.2f}")
        self.logger.info(f"Latency (mean):     {metrics['latency_mean_ms']:.2f} ms")
        self.logger.info(f"Latency (std):      {metrics['latency_std_ms']:.2f} ms")
        self.logger.info(f"Latency (min):      {metrics['latency_min_ms']:.2f} ms")
        self.logger.info(f"Latency (max):      {metrics['latency_max_ms']:.2f} ms")
        self.logger.info(f"Latency (p95):      {metrics['latency_p95_ms']:.2f} ms")
        self.logger.info(f"Latency (p99):      {metrics['latency_p99_ms']:.2f} ms")
        self.logger.info("-"*80)
        
        if 'gpu_memory_allocated_mb' in metrics:
            self.logger.info(f"GPU Memory (alloc): {metrics['gpu_memory_allocated_mb']:.2f} MB")
            self.logger.info(f"GPU Memory (res):   {metrics['gpu_memory_reserved_mb']:.2f} MB")
        
        self.logger.info(f"CPU Memory:         {metrics['cpu_memory_mb']:.2f} MB")
        self.logger.info(f"Model Size:         {metrics['model_size_mb']:.2f} MB")
        self.logger.info(f"Parameters:         {metrics['num_parameters']:,}")
        self.logger.info("="*80)
    
    def save_metrics(self, metrics: Dict[str, float], output_path: str):
        """Save metrics to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved: {output_path}")
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        input_size: tuple = (1, 3, 640, 640),
        num_iterations: int = 100
    ):
        """
        Compare multiple models
        
        Args:
            models: Dict of model_name -> model
            input_size: Input size
            num_iterations: Iterations per model
        """
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"\nBenchmarking: {name}")
            metrics = self.benchmark_model(model, input_size, num_iterations)
            results[name] = metrics
        
        # Print comparison
        self.logger.info("\n" + "="*80)
        self.logger.info("Model Comparison")
        self.logger.info("="*80)
        
        for name, metrics in results.items():
            self.logger.info(f"\n{name}:")
            self.logger.info(f"  FPS: {metrics['fps']:.2f}")
            self.logger.info(f"  Latency: {metrics['latency_mean_ms']:.2f} ms")
            self.logger.info(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
        
        self.logger.info("="*80)
        
        return results


if __name__ == "__main__":
    # Example
    benchmarker = PerformanceBenchmarker(device="cpu")
    
    # Dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1)
    )
    
    metrics = benchmarker.benchmark_model(model, num_iterations=50)
    benchmarker.print_metrics(metrics)
