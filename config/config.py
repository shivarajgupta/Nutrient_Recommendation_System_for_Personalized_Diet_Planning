import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent

# Database configuration
DATABASE_CONFIG = {
    'name': 'nutriusher.db',
    'path': BASE_DIR / 'data'
}

# Application settings
APP_CONFIG = {
    'name': 'NutriUsher',
    'description': 'Personalized Diet Planning Made Easy',
    'theme_color': '#46A017'
}

# Static files
STATIC_DIR = BASE_DIR / 'static'
IMAGES_DIR = STATIC_DIR / 'images'
STYLES_DIR = STATIC_DIR / 'styles'