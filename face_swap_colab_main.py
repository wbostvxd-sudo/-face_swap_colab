#!/usr/bin/env python3

import os
import sys

# Set environment variables for Colab optimization
os.environ['OMP_NUM_THREADS'] = '1'

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import core

if __name__ == '__main__':
    # If no arguments are provided, default to 'run' to launch the UI
    if len(sys.argv) == 1:
        sys.argv.append('run')
    
    core.cli()
