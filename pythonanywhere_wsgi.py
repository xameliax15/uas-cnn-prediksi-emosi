
# This file contains the WSGI configuration required to serve up your
# web application at http://<your-username>.pythonanywhere.com/
# It works by setting the variable 'application' to a WSGI handler of some
# description.

import sys
import os

# The path to your project directory
# ADJUST THIS PATH to match where you uploaded your files
# Example: If you uploaded to /home/yourusername/mysite/
project_home = os.path.expanduser('~/mysite')
if project_home not in sys.path:
    sys.path.append(project_home)

# Backend folder needs to be in path too
backend_path = os.path.join(project_home, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Import Flask app
from backend.main import app as application
