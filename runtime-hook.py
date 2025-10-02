
import os
import sys
if getattr(sys, 'frozen', False):
    os.environ['QT_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')
    os.environ['PATH'] = os.path.join(sys._MEIPASS, 'torch', 'lib') + ';' + \
                        os.path.join(sys._MEIPASS, 'numpy', '.libs') + ';' + \
                        os.path.join(sys._MEIPASS, 'pandas', '_libs') + ';' + \
                        os.environ['PATH']
