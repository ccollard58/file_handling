# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files for various packages
datas = []

# Collect hidden imports
hiddenimports = [
    'PyQt6.sip',
    'PyQt6.QtPdf',
    'PIL._tkinter_finder',
    'PIL.Image',
    'langchain_community.vectorstores',
    'langchain_community.embeddings',
    'langchain_ollama',
    'pytesseract',
    'cv2',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'plotly',
    'networkx',
    'nltk',
    'sklearn',
    'scipy',
    'torch',
    'torchvision',
    'paddleocr',
    'easyocr',
    'pdf2image',
    'PyMuPDF',
    'pypdf',
    'python_docx',
    'openpyxl',
    'requests',
    'ollama',
    'pywin32',
    'win32api',
    'win32gui',
    'win32con',
    'pywintypes',
    'pkg_resources.py2_warn',
    'pkg_resources.markers',
]

# Collect all submodules for langchain packages
hiddenimports += collect_submodules('langchain')
hiddenimports += collect_submodules('langchain_community')
hiddenimports += collect_submodules('langchain_ollama')
hiddenimports += collect_submodules('langchain_core')

# Add additional hidden imports for PyQt6
hiddenimports += [
    'PyQt6.QtCore',
    'PyQt6.QtGui', 
    'PyQt6.QtWidgets',
    'PyQt6.QtPdf',
]

# Analysis configuration
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'tk',
        '_tkinter',
        'turtle',
        'matplotlib.backends._backend_tk',
    ],
    noarchive=False,
    optimize=0,
)

# Remove duplicate entries
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DocumentOrganizer_Portable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
