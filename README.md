# DeepLearningNetworkManagement
EMS-EMC

Contined are all the files and programs created for our EMC project, to use the models follow the instructions below.

## Installation
Our project uses the python tensorflow package, which requires a 64-bit installation of python 3.7 or lower
```bash
pip install tensorflow
```
## Usage
To load the model into a python script simply use the commands below
```python
from tensorflow.keras.models import load_model

model = load_model(filePath)
```
