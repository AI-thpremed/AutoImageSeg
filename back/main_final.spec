# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# ====================== 基础配置 ======================
sys.setrecursionlimit(5000)
env_dir = r'D:\Anaconda3\envs\pytorch_gpu'
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin'

# ====================== 1. 生成必要运行时文件 ======================
# 生成 qt.conf
qt_conf_content = """[Paths]
Prefix = PySide6
Plugins = PySide6/plugins
"""
qt_conf_path = os.path.join(os.getcwd(), 'qt.conf')
if not os.path.exists(qt_conf_path):
    with open(qt_conf_path, 'w', encoding='utf-8') as f:
        f.write(qt_conf_content)

# 生成 UTF-8 修复脚本
utf8_fix_path = os.path.join(os.getcwd(), '_utf8_fix.py')
with open(utf8_fix_path, 'w', encoding='utf-8') as f:
    f.write("""
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
""")

# 生成 runtime-hook.py
runtime_hook_path = os.path.join(os.getcwd(), 'runtime-hook.py')
with open(runtime_hook_path, 'w', encoding='utf-8') as f:
    f.write("""
import os
import sys
if getattr(sys, 'frozen', False):
    os.environ['QT_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')
    os.environ['PATH'] = os.path.join(sys._MEIPASS, 'torch', 'lib') + ';' + \\
                        os.path.join(sys._MEIPASS, 'numpy', '.libs') + ';' + \\
                        os.path.join(sys._MEIPASS, 'pandas', '_libs') + ';' + \\
                        os.environ['PATH']
""")

# ====================== 2. 收集依赖 ======================
# 强制收集 Pandas 所有依赖
pandas_datas, pandas_binaries, pandas_hidden = collect_all('pandas')

# ====================== 3. 分析配置 ======================
a = Analysis(
    ['main.py', utf8_fix_path],
    pathex=[os.getcwd(), env_dir],
	binaries=[
        # Python 核心
        (f'{env_dir}\\python39.dll', '.'),
        
        # PySide6
        (f'{env_dir}\\Lib\\site-packages\\shiboken6\\*.dll', '.'),
        (f'{env_dir}\\Lib\\site-packages\\PySide6\\plugins\\platforms\\*.dll', 'PySide6/plugins/platforms'),
        
        # PyTorch
        (f'{env_dir}\\Lib\\site-packages\\torch\\lib\\*.dll', 'torch/lib'),
        
        # CUDA
        (f'{cuda_path}\\cudart64_*.dll', '.'),
        (f'{cuda_path}\\cublas64_*.dll', '.'),
        (f'{cuda_path}\\cudnn64_*.dll', '.'),
        

        # 手动添加 Pandas 二进制文件（确保覆盖所有）
        (f'{env_dir}\\Lib\\site-packages\\pandas\\_libs\\*.pyd', 'pandas/_libs'),
        (f'{env_dir}\\Lib\\site-packages\\pandas\\_libs\\window\\*.pyd', 'pandas/_libs/window'),
        (f'{env_dir}\\Lib\\site-packages\\pandas\\_libs\\tslibs\\*.pyd', 'pandas/_libs/tslibs'),
        
        # NumPy
        (f'{env_dir}\\Lib\\site-packages\\numpy\\.libs\\*.dll', 'numpy/.libs'),
    ],
    datas=[
        # 配置文件
        (qt_conf_path, '.'),
        
        # Python 标准库
        (f'{env_dir}\\Lib\\encodings', 'Lib/encodings'),
        
        # PyTorch
        (f'{env_dir}\\Lib\\site-packages\\torch\\**', 'torch'),
        (f'{env_dir}\\Lib\\site-packages\\torchvision\\**', 'torchvision'),
        
        # 其他资源
        ('introduction.ui', '.'),
        
        # 添加 Pandas 数据文件
		(r'D:\Anaconda3\envs\pytorch_gpu\Lib\site-packages\pandas\_libs\aggregations.cp39-win_amd64.pyd', '_internal/pandas/_libs'),
    ],
    hiddenimports=[
        # Pandas
        'pandas._libs.aggregations',
        'pandas._libs.tslibs',
        'pandas._libs.window.aggregations',
        'pandas._libs.join',
        'pandas._libs.groupby',
        
        # PyTorch
        'torch._C',
        'torch._dynamo',
        'torch.autograd',
        
        # NumPy
        'numpy.core._multiarray_umath',
        
        # 编码
        'encodings',
        'encodings.aliases',
        'encodings.utf_8',
        
        # PySide6
        'shiboken6',
        'PySide6.QtCore',
        
        # 添加 Pandas 隐藏依赖
        *pandas_hidden,
    ],
    hookspath=['hooks'],  # 如果有自定义 hook 目录
    runtime_hooks=[runtime_hook_path],
    excludes=['PyQt5'],
    noarchive=False,
)

# ====================== 4. 构建配置 ======================
pyz = PYZ(a.pure, a.zipped_data)
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