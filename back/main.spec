# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# 环境配置
env_dir = r'd:\Anaconda3\envs\pytorch_gpu'
python_dir = sys.prefix

# 确保qt.conf存在
qt_conf_path = os.path.join(env_dir, 'Lib', 'site-packages', 'PySide6', 'qt.conf')
if not os.path.exists(qt_conf_path):
    with open('qt.conf', 'w', encoding='utf-8') as f:
        f.write("""[Paths]
Prefix = PySide6
Plugins = PySide6/plugins
""")
    qt_conf_path = 'qt.conf'

# 自动收集PySide6相关文件
pyside6_data = collect_data_files('PySide6', include_py_files=True)
pyside6_libs = collect_dynamic_libs('PySide6')

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[r'D:\pywork\segment-test\test', env_dir],
    binaries=[
        (f'{env_dir}\\python39.dll', '.'),
        *[(lib[0], '.') for lib in pyside6_libs],  # PySide6和Shiboken6的DLL
        (f'{env_dir}\\Lib\\site-packages\\torch\\lib\\*.dll', 'torch/lib'),
        (f'{env_dir}\\Lib\\site-packages\\pandas\\_libs\\*.pyd', 'pandas/_libs'),
        (f'{env_dir}\\Lib\\site-packages\\numpy\\.libs\\*.dll', 'numpy/.libs'),
    ],
    datas=[
        *pyside6_data,  # PySide6的资源和插件
        (qt_conf_path, '.'),
        (f'{env_dir}\\Lib\\encodings', 'Lib/encodings'),
        (f'{env_dir}\\Lib\\site-packages\\torch\\**', 'torch'),
        (f'{env_dir}\\Lib\\site-packages\\torchvision\\**', 'torchvision'),
        ('introduction.ui', '.'),  # 确保UI文件被打包
    ],
    hiddenimports=[
        'shiboken6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'torch',
        'torch._C',
        'torchvision',
        'pandas._libs.aggregations',
        'pandas._libs.tslibs',
        'numpy',
        'encodings',
        'encodings.aliases',
        'encodings.ascii',
        'encodings.utf_8',
        'config_manager',  # 确保您的配置管理器被包含
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_pyside6.py'],
    excludes=['PyQt5'],
    noarchive=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 改为False隐藏控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    optimize=1
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main'
)