"""
Model Pruning
Optimization techniques for model size reduction including structured and unstructured pruning.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Union

class ModelPruning:
    """Utilities for model pruning to reduce size and inference time"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def prune_global(self, amount: float = 0.2, method: str = 'l1'):
        """
        Apply global unstructured pruning to all Conv2d and Linear layers.
        
        Args:
            amount: Fraction of connections to prune (0.0 to 1.0)
            method: Pruning method 'l1' or 'random'
        """
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
                
        if not parameters_to_prune:
            print("No prune-able layers found")
            return
            
        print(f"Global pruning: {len(parameters_to_prune)} layers, amount={amount}")
        
        pruning_method = prune.L1Unstructured if method == 'l1' else prune.RandomUnstructured
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=amount,
        )
        
    def prune_structured(self, amount: float = 0.2, amount_dim: int = 0):
        """
        Apply structured pruning (e.g., channel pruning)
        
        Args:
            amount: Fraction of channels to prune
            amount_dim: Dimension to prune (0=output channels, 1=input channels)
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=amount_dim)
                
    def make_permanent(self):
        """Remove pruning re-parameterization to make changes permanent"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                 if prune.is_pruned(module):
                     prune.remove(module, 'weight')
                     
    def get_sparsity(self):
        """Calculate global sparsity"""
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += torch.sum(param == 0).item()
                
        return zero_params / total_params if total_params > 0 else 0.0
