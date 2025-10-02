# -*- mode: python ; coding: utf-8 -*-
import sys
import os
sys.setrecursionlimit(5000)

# 1. 自动生成 qt.conf 文件（如果不存在）
qt_conf_content = """[Paths]
Prefix = PySide6
Plugins = PySide6/plugins
"""
qt_conf_path = os.path.join(os.getcwd(), 'qt.conf')
if not os.path.exists(qt_conf_path):
    with open(qt_conf_path, 'w', encoding='utf-8') as f:
        f.write(qt_conf_content)

# 2. 创建 UTF-8 修复脚本
utf8_fix_path = os.path.join(os.getcwd(), '_utf8_fix.py')
with open(utf8_fix_path, 'w', encoding='utf-8') as f:
    f.write("""
import sys
import io
import os

# 强制 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置 Qt 插件路径
if hasattr(sys, '_MEIPASS'):
    os.environ['QT_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')
""")

block_cipher = None

env_dir = r'd:\Anaconda3\envs\pytorch_gpu'

# 3. 分析阶段
a = Analysis(
    ['main.py', utf8_fix_path],  # 包含 UTF-8 修复脚本
    pathex=[r'D:\pywork\segment-test\test', env_dir],
    binaries=[
        (f'{env_dir}\\Lib\\site-packages\\shiboken6\\*.dll', '.'),
        (f'{env_dir}\\python39.dll', '.'),
        (f'{env_dir}\\Lib\\site-packages\\pandas\\_libs\\*.pyd', 'pandas/_libs'),
        (f'{env_dir}\\Lib\\site-packages\\numpy\\.libs\\*.dll', 'numpy/.libs'),
        (f'{env_dir}\\Lib\\site-packages\\PySide6\\plugins\\platforms\\*.dll', 'PySide6/plugins/platforms'),
    ],
    datas=[
        (qt_conf_path, '.'),  # 使用生成的 qt.conf
        (f'{env_dir}\\Lib\\encodings', 'Lib/encodings'),
        (f'{env_dir}\\Lib\\site-packages\\torch\\**', 'torch'),
        (f'{env_dir}\\Lib\\site-packages\\torchvision\\**', 'torchvision'),
        ('introduction.ui', '.'),
    ],
    hiddenimports=[
        'pandas._libs.aggregations',
        'pandas._libs.tslibs',
        'numpy',
        'encodings',
        'encodings.aliases',
        'encodings.ascii',
        'encodings.utf_8',
        'shiboken6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
)

# 4. 生成 PYZ 归档
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 5. 生成 EXE 可执行文件
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    optimize=1
)

# 6. 收集所有文件
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main'
)