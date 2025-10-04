Convert .ui Files to .py Files
pyside6-uic train_window_mask.ui > ui_train_window_mask.py
pyside6-uic train_window.ui -o ui_train_window.py
pyside6-uic eval.ui -o ui_eval.py
pyside6-uic post_window.ui -o ui_post_window.py


These generated .py files should be in UTF-8 encoding.