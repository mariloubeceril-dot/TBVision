# TBVision

TBVision is an AI-assisted screening tool that analyzes chest X-ray images to detect possible tuberculosis cases.

## Installation

1. Install Python
[https://www.python.org/downloads/
](https://www.python.org/downloads/release/python-3100/)

3. Install required libraries

pip install -r requirements.txt

3. Download the trained model

[https://drive.google.com/drive/u/2/search?q=.h5
](https://drive.google.com/file/d/1k1utFGp9qhwjUN1L8ISxZRWPNJ_x65F6/view?usp=sharing)

Place the file `TBVision_DenseNet_Model.h5` in the same folder as `tbvision_app.py`.

4. Run TBVision


pyinstaller app.py ^ --onefile 
--collect-data gradio
--collect-data gradio_client
 --collect-all gradio
pyinstaller --onedir --collect-all gradio --collect-all gradio_client --collect-all tensorflow --collect-all keras --collect-all safetensors --add-data "TBVision_DenseNet_Model.h5;." app.py
doskey /history > commands.txt
