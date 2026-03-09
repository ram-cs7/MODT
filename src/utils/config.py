"""
Configuration Manager for Military Object Detection System
Handles loading, merging, and validating configuration files
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator


class ConfigManager:
    """Configuration manager with hierarchical config support"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file. If None, loads default config
        """
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.default_config_path = self.config_dir / "default.yaml"
        
        # Load default config
        self.config = self.load_yaml(self.default_config_path)
        
        # Load and merge custom config if provided
        if config_path:
            custom_config = self.load_yaml(config_path)
            self.config = self.merge_configs(self.config, custom_config)
    
    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """
        Recursively merge two configuration dictionaries
        Override values take precedence over base values
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('model.detector.type')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        Example: config.set('model.detector.type', 'yolov8')
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Union[str, Path]):
        """Save current configuration to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-like assignment"""
        self.set(key, value)
    
    def __repr__(self) -> str:
        return f"ConfigManager({len(self.config)} top-level keys)"


def load_config(config_name: str = "default") -> ConfigManager:
    """
    Convenience function to load configuration
    
    Args:
        config_name: Name of config file (without .yaml extension)
                    or full path to config file
    
    Returns:
        ConfigManager instance
    """
    config_dir = Path(__file__).parent.parent.parent / "config"
    
    # Check if it's a full path
    if Path(config_name).exists():
        return ConfigManager(config_name)
    
    # Check if it's a config name
    config_path = config_dir / f"{config_name}.yaml"
    if config_path.exists():
        return ConfigManager(config_path)
    
    # Default to default.yaml
    return ConfigManager()


if __name__ == "__main__":
    # Example usage
    config = load_config("default")
    
    print("Model type:", config.get("model.detector.type"))
    print("Input size:", config.get("model.detector.input_size"))
    print("Batch size:", config.get("training.batch_size"))
    
    # Set new value
    config.set("training.batch_size", 32)
    print("New batch size:", config.get("training.batch_size"))
    
    # Save modified config
    # config.save("./config/custom.yaml")
