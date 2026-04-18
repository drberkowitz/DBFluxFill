"""
DBFluxFill
FLUX.1 Fill Dev for Nuke.
AI-powered inpainting using Black Forest Labs FLUX Fill.

Installation:
    1. Place the DBFluxFill folder in your .nuke directory
    2. Run setup.bat to configure and download models
    3. Add to your init.py:
           nuke.pluginAddPath('./DBFluxFill')
           import DBFluxFill

Version: 1.0.0
"""

import sys
import os
import nuke

# Version check - warn if running on unsupported Python
if sys.version_info < (3, 7):
    print("DBFluxFill: Warning - Python 3.7 or higher required. "
          "Current version: {}.{}.{}".format(*sys.version_info[:3]))

__version__ = "1.0.0"