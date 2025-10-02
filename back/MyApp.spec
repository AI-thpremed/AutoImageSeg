# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files

# ====================== 强制单文件夹模式 ======================
a = Analysis(
    ['main.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=[
        ('introduction.ui', '.'),       # UI文件
        ('images/*', 'images'),         # 图片文件夹
        *collect_data_files('PySide6', subdir='plugins')  # Qt插件
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'numpy.core._multiarray_umath'
    ],
    hookspath=[],
    excludes=['torch', 'torchvision'],
    noarchive=False
)

# ====================== 生成清晰目录结构 ======================
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='AutoImageSeg',
    debug=False,
    strip=False,
    upx=True,
    console=False,  # 设为True可查看错误信息
    disable_windowed_traceback=False
)

# 关键修改：移除COLLECT以生成单层结构