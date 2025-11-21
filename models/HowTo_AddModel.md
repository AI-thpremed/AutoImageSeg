# How to Add a New Model

## Step 1: Create Model File
Place your model file under the models directory:
models/your_net.py


## Step 2: Implement Constructor
Use the following constructor signature in your model class:

def __init__(self, in_ch: int = 3, out_ch: int = 4):


in_ch: Always 3 (for RGB input channels)

out_ch: Automatically read from configuration file at runtime

## Step 3: Register Model in Workers
Add your model to these worker files:

File: train_worker.py
File: train_worker_mask.py
File: infer_worker.py

# Add import statement
from models.your_net import YourNet

# Add to model dictionary
model_map['YourNet'] = YourNet

for example:

        model_map = {'UNet_B': UNet_B,'UNet_M': UNet_M,'UNet_S': UNet_S,'UNet_T': UNet_T, 'MobileUNet': MobileUNet,
                     'FastSCNN': FastSCNN, 'UNext': UNext,'AttU_Net': AttU_Net,
                     'NestedUNet': NestedUNet,'UNetResnet': UNetResnet,
                     'FCN': FCN,'LinkNet': LinkNet}


## Step 4: Update GUI Interface
Part A: Edit UI Files
Open these files in QtDesigner:

train_window.ui

train_window_mask.ui

Part B: Add ComboBox Item
Locate the combo_algo combobox and insert:
<item>
    <property name="text">
        <string>YourNet</string>
    </property>
</item>

Part C: Regenerate UI Code
Run these commands in terminal:

pyside6-uic train_window.ui -o ui_train_window.py
pyside6-uic train_window_mask.ui -o ui_train_window_mask.py


## Step 5: Final Steps
Choose one of these options:

1,Rebuild the executable package, or


2,Run the software from source code

Your new model "YourNet" will now appear in the algorithm selection dropdown menu.




## Rebuild the Software

use nuitka package to build the exe Software

CPU version:
 python -m nuitka --standalone --msvc=latest --enable-plugin=pyside6 --include-data-dir=G:\miniconda3\envs\pytorch_gpu\Lib\site-packages\PySide6\plugins=PySide6\qt-plugins --include-data-dir=G:\miniconda3\envs\pytorch_gpu\Lib\site-packages\torch\lib=torch\lib --output-dir=dist --windows-disable-console main.py


GPU version:
python -m nuitka --standalone --msvc=latest --enable-plugin=pyside6 --include-data-dir=G:\miniconda3\envs\pytorch_cpu\Lib\site-packages\PySide6\plugins=PySide6\qt-plugins --include-data-dir=G:\miniconda3\envs\pytorch_cpu\Lib\site-packages\torch\lib=torch\lib --include-data-file=introduction.ui=introduction.ui --output-dir=dist --windows-disable-console main.py

