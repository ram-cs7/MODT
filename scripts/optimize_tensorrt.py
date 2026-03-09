#!/usr/bin/env python3
"""
TensorRT Optimization Script for Military Object Detection
Optimizes ONNX models for Jetson Edge Deployment
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import torch

def setup_logger():
    """Setup basic logger"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("TRT-Optimizer")

def check_trtexec():
    """Check if trtexec is available"""
    try:
        subprocess.run(['trtexec', '--help'], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False

def optimize_model(onnx_path, output_path, precision='fp16', workspace=3072):
    """
    Optimize ONNX model using trtexec
    """
    logger = setup_logger()
    
    if not check_trtexec():
        logger.warning("trtexec not found! This script requires TensorRT to be installed.")
        logger.warning("If running on Windows PC without TensorRT, this step is skipped.")
        return False
        
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}',
        f'--memPoolSize=workspace:{workspace}',
        '--verbose'
    ]
    
    if precision == 'fp16':
        cmd.append('--fp16')
    elif precision == 'int8':
        cmd.append('--int8')
        
    logger.info(f"Starting TensorRT optimization: {onnx_path} -> {output_path}")
    logger.info(f"Precision: {precision}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                
        if process.returncode == 0:
            logger.info("Optimization successful!")
            return True
        else:
            logger.error("Optimization failed.")
            return False
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Optimize model for TensorRT')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--output', type=str, default=None, help='Output engine path')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'], help='Precision mode')
    parser.add_argument('--workspace', type=int, default=4096, help='Workspace size in MB')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.onnx).with_suffix('.engine'))
        
    optimize_model(args.onnx, args.output, args.precision, args.workspace)

if __name__ == "__main__":
    main()
