"""
CUDA Utilities
Auto-detects and switches to GPU (CuPy) or falls back to CPU (NumPy).
"""

import os

# Try to detect CUDA and use CuPy, otherwise fallback to NumPy
def get_array_module():
    """
    Returns the appropriate array module:
    - CuPy if CUDA is available
    - NumPy as fallback
    """
    # Check environment variable for override
    if os.environ.get('FORCE_CPU', '').lower() == 'true':
        import numpy as np
        print("Using NumPy (CPU) - FORCE_CPU enabled")
        return np
    
    # Try CuPy with CUDA
    try:
        import cupy as cp
        if cp.cuda.is_available():
            print(f"Using CuPy (GPU) - CUDA device: {cp.cuda.Device()}")
            return cp
    except ImportError:
        pass
    except Exception as e:
        print(f"CuPy available but CUDA error: {e}")
    
    # Fallback to NumPy
    import numpy as np
    print("Using NumPy (CPU) - CUDA not available")
    return np


def get_np():
    """Shortcut for get_array_module()"""
    return get_array_module()


# Auto-initialize on import
np = get_array_module()

__all__ = ['np', 'get_array_module', 'get_np']