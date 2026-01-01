"""
SAM3 FlatBug Configuration

This configuration file mirrors the FlatBug repository's default configuration
to ensure a fair comparison between SAM3 and FlatBug models.

Reference: /home/dolma/repo/flat-bug/src/flat_bug/config.py
"""

from typing import Tuple, Dict, Any
from collections import OrderedDict
import os
import yaml

# ==========================
# CONFIGURATION PARAMETERS
# ==========================

CFG_PARAMS = [
    "SCORE_THRESHOLD",
    "IOU_THRESHOLD",
    "MINIMUM_TILE_OVERLAP",
    "EDGE_CASE_MARGIN",
    "IMAGE_BOUNDARY_MARGIN",
    "MIN_MAX_OBJ_SIZE",
    "MAX_MASK_SIZE",
    "PREFER_POLYGONS",
    "TILE_SIZE",
    "SCALE_INCREMENT",
    "PADDING",
    "PROMPT_PLURAL",
    "PROMPT_SINGULAR",
    "CATEGORY_ID",
]

CFG_DESCRIPTION = {
    "SCORE_THRESHOLD": "Minimum confidence score for a prediction to be considered valid.",
    "IOU_THRESHOLD": "IoU threshold for NMS - predictions with IoU above this are considered duplicates.",
    "MINIMUM_TILE_OVERLAP": "Minimum overlap (in pixels) between adjacent tiles when splitting the image.",
    "EDGE_CASE_MARGIN": "Margin (in pixels) for edge cases. Detections within this margin of tile edges are filtered.",
    "IMAGE_BOUNDARY_MARGIN": "Margin (in pixels) from actual image boundaries. Detections touching these edges are removed as likely truncated.",
    "MIN_MAX_OBJ_SIZE": "Min and max object size as sqrt(bbox_area). Objects outside this range are filtered.",
    "MAX_MASK_SIZE": "Maximum mask resolution. May affect precision if exceeded.",
    "PREFER_POLYGONS": "If True, convert masks to polygons for compact representation.",
    "TILE_SIZE": "Fixed tile size (1024 for SAM3 optimal input). Do not change unless necessary.",
    "SCALE_INCREMENT": "Scale factor between pyramid levels (2/3 means each level is 1.5x larger than previous).",
    "PADDING": "Padding (in pixels) added to image borders before inference to handle edge detections.",
    "PROMPT_PLURAL": "Text prompt for tiles (expecting multiple objects).",
    "PROMPT_SINGULAR": "Text prompt for global/zoomed-out view (expecting larger objects).",
    "CATEGORY_ID": "COCO category ID for output annotations.",
}

# ==========================
# DEFAULT CONFIGURATION
# Matches FlatBug defaults for fair comparison
# ==========================

DEFAULT_CFG: Dict[str, Any] = {
    # --- Core Detection Thresholds (from FlatBug) ---
    "SCORE_THRESHOLD": 0.25,         # Minimum confidence for a prediction
    "IOU_THRESHOLD": 0.20,           # IoU threshold for NMS (aggressive duplicate removal)
    "MINIMUM_TILE_OVERLAP": 384,     # Pixels overlap between adjacent tiles
    "EDGE_CASE_MARGIN": 16,          # Margin to filter edge-case detections at tile boundaries
    "IMAGE_BOUNDARY_MARGIN": 10,     # Margin to filter detections at actual image edges (removes truncated objects)
    "MIN_MAX_OBJ_SIZE": (32, 10**8), # Min/max object size (sqrt of bbox area)
    "MAX_MASK_SIZE": 1024,           # Maximum mask resolution
    "PREFER_POLYGONS": True,         # Use polygons instead of masks
    
    # --- Tiling & Pyramid (from FlatBug) ---
    "TILE_SIZE": 1024,               # Fixed tile size (model architecture)
    "SCALE_INCREMENT": 2/3,          # Pyramid ratio: 1.0, 0.66, 0.44, ...
    "PADDING": 32,                   # Padding added to image borders (2x EDGE_CASE_MARGIN)
    
    # --- SAM3-Specific ---
    "PROMPT_PLURAL": "insects",      # Prompt for high-res tiles
    "PROMPT_SINGULAR": "insect",     # Prompt for global view
    "CATEGORY_ID": 1,                # COCO category ID
}

# Legacy configuration (for reference - matches older FlatBug versions)
LEGACY_CFG: Dict[str, Any] = {
    "SCORE_THRESHOLD": 0.5,
    "IOU_THRESHOLD": 0.5,
    "MINIMUM_TILE_OVERLAP": 256,
    "EDGE_CASE_MARGIN": 128,
    "IMAGE_BOUNDARY_MARGIN": 10,
    "MIN_MAX_OBJ_SIZE": (16, 1024),
    "MAX_MASK_SIZE": 1024,
    "PREFER_POLYGONS": True,
    "TILE_SIZE": 1024,
    "SCALE_INCREMENT": 2/3,
    "PADDING": 32,
    "PROMPT_PLURAL": "insects",
    "PROMPT_SINGULAR": "insect",
    "CATEGORY_ID": 1,
}


def validate_cfg(cfg: Dict[str, Any], strict: bool = False) -> bool:
    """
    Validate configuration dictionary types and values.
    
    Args:
        cfg: Configuration dictionary to validate
        strict: If True, raise error on unknown keys
        
    Returns:
        True if validation passes
    """
    type_checks = {
        "SCORE_THRESHOLD": (float, int),
        "IOU_THRESHOLD": (float, int),
        "MINIMUM_TILE_OVERLAP": int,
        "EDGE_CASE_MARGIN": int,
        "IMAGE_BOUNDARY_MARGIN": int,
        "MIN_MAX_OBJ_SIZE": (tuple, list),
        "MAX_MASK_SIZE": int,
        "PREFER_POLYGONS": bool,
        "TILE_SIZE": int,
        "SCALE_INCREMENT": (float, int),
        "PADDING": int,
        "PROMPT_PLURAL": str,
        "PROMPT_SINGULAR": str,
        "CATEGORY_ID": int,
    }
    
    for key, value in cfg.items():
        if key not in type_checks:
            if strict:
                raise KeyError(f"Unknown config parameter: {key}")
            continue
            
        expected_type = type_checks[key]
        if not isinstance(value, expected_type):
            raise TypeError(f"Config '{key}' expected {expected_type}, got {type(value)}")
    
    # Value range checks
    if not (0 <= cfg.get("SCORE_THRESHOLD", 0.2) <= 1):
        raise ValueError("SCORE_THRESHOLD must be between 0 and 1")
    if not (0 <= cfg.get("IOU_THRESHOLD", 0.2) <= 1):
        raise ValueError("IOU_THRESHOLD must be between 0 and 1")
    if cfg.get("TILE_SIZE", 1024) <= 0:
        raise ValueError("TILE_SIZE must be positive")
    if not (0 < cfg.get("SCALE_INCREMENT", 2/3) < 1):
        raise ValueError("SCALE_INCREMENT must be between 0 and 1")
        
    return True


def read_cfg(path: str, strict: bool = False) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Missing keys are filled with defaults.
    
    Args:
        path: Path to YAML config file
        strict: If True, raise error on unknown keys
        
    Returns:
        Complete configuration dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise ValueError(f"Config must be a YAML file, got: {path}")
    
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Validate
    validate_cfg(cfg, strict)
    
    # Fill missing keys with defaults
    for key in DEFAULT_CFG:
        if key not in cfg:
            cfg[key] = DEFAULT_CFG[key]
    
    return cfg


def write_cfg(cfg: Dict[str, Any], path: str, overwrite: bool = False) -> str:
    """
    Save configuration to YAML file.
    
    Args:
        cfg: Configuration dictionary
        path: Output path for YAML file
        overwrite: If True, overwrite existing file
        
    Returns:
        Path to saved config file
    """
    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise ValueError(f"Config must be a YAML file, got: {path}")
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"Config file already exists: {path}")
    
    validate_cfg(cfg)
    
    # Create ordered dict for pretty output
    sorted_cfg = OrderedDict()
    for key in CFG_PARAMS:
        if key in cfg:
            # Convert tuples to lists for YAML
            value = cfg[key]
            if isinstance(value, tuple):
                value = list(value)
            sorted_cfg[key] = value
    
    # Add any extra keys
    for key in cfg:
        if key not in sorted_cfg:
            sorted_cfg[key] = cfg[key]
    
    with open(path, "w") as f:
        yaml.safe_dump(dict(sorted_cfg), f, sort_keys=False, default_flow_style=None)
    
    return path


def get_cfg(cfg_input=None) -> Dict[str, Any]:
    """
    Get configuration from various input types.
    
    Args:
        cfg_input: Can be None (use defaults), dict, or path to YAML file
        
    Returns:
        Complete configuration dictionary
    """
    if cfg_input is None:
        return DEFAULT_CFG.copy()
    elif isinstance(cfg_input, dict):
        cfg = DEFAULT_CFG.copy()
        cfg.update(cfg_input)
        validate_cfg(cfg)
        return cfg
    elif isinstance(cfg_input, str):
        return read_cfg(cfg_input)
    else:
        raise TypeError(f"cfg must be None, dict, or path string, got {type(cfg_input)}")


def print_cfg(cfg: Dict[str, Any] = None):
    """Print configuration in a readable format."""
    if cfg is None:
        cfg = DEFAULT_CFG
    
    print("=" * 60)
    print("SAM3 FlatBug Configuration")
    print("=" * 60)
    for key in CFG_PARAMS:
        if key in cfg:
            desc = CFG_DESCRIPTION.get(key, "")
            print(f"  {key}: {cfg[key]}")
            if desc:
                print(f"      -> {desc}")
    print("=" * 60)


if __name__ == "__main__":
    print_cfg()
