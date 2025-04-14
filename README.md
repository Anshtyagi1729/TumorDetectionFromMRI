# TumorDetectionFromMRI
this is a cnn trained to classify tumors in brain like glioma , meningioma , pituitory and so on from the MRI images


## Download the Model

The model file `model.pth` can be downloaded from [Google Drive link](https://drive.google.com/file/d/1_jiv89Oo0Tvz6E76Ol6jT6I0CQ4txYOI/view?usp=sharing).
this is the file containing the weights of the CNN that i trained 
Once downloaded, place it in the root directory of the project and use the following code to load it:

```python
import torch
from app.model import load_model

model = load_model('path_to_downloaded_model/model.pth')
