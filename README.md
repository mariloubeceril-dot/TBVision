# TBVision

TBVision is an AI-assisted screening tool that analyzes chest X-ray images to detect possible tuberculosis cases.

## Installation

1. Install Python
[https://www.python.org/downloads/
](https://www.python.org/downloads/release/python-3100/)

3. Install required libraries
```bash
pip install -r requirements.txt
```

3. Download the trained model

[https://drive.google.com/drive/u/2/search?q=.h5
](https://drive.google.com/file/d/1k1utFGp9qhwjUN1L8ISxZRWPNJ_x65F6/view?usp=sharing)

Place the file `TBVision_DenseNet_Model.h5` in the same folder as `tbvision_app.py`.

4. Run TBVision
## Development Commands

```bash
pyinstaller app.py ^ --onefile 
--collect-data gradio
--collect-data gradio_client
 --collect-all gradio
pyinstaller --onedir --collect-all gradio --collect-all gradio_client --collect-all tensorflow --collect-all keras --collect-all safetensors --add-data "TBVision_DenseNet_Model.h5;." app.py
doskey /history > commands.txt
```

### PyInstaller Confirmation

After running the commands, PyInstaller may display the following warning:

```text
WARNING: The output directory "C:\Users\Admin\Desktop\TBVision_Offline\dist\app" and ALL ITS CONTENTS will be REMOVED! Continue? (y/N)
```

Delete the files from the previous build if necessary, then type:

```text
y
```

It should look like this in the terminal:

```text
WARNING: The output directory "C:\Users\Admin\Desktop\TBVision_Offline\dist\app" and ALL ITS CONTENTS will be REMOVED! Continue? (y/N)y
```


5.

```bash
python app.py
```

### Expected Output

After running the command, the terminal will display messages similar to the following:

```text
C:\Users\Admin\Desktop\TBVision_Offline>python app.py
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 tensorflow/core/util/port.cc:153 oneDNN custom operations are on.
I0000 tensorflow/core/platform/cpu_feature_guard.cc:210 This TensorFlow binary is optimized to use available CPU instructions.

C:\Users\Admin\Desktop\TBVision_Offline\app.py:204: UserWarning: The parameters have been moved from the Blocks constructor to the launch() method in Gradio 6.0: theme.

* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

Open the following link in your browser:

```
http://127.0.0.1:7860
```




