# Kalchas

A greek polytonic OCR library  


Kalchas is a greek polytonic OCR library. 


### Installation 

TODO


###  Example usage: 



```python

import numpy as np
from PIL import Image
from kraken import binarization

from kalchas.ocr import lists_models, load_ocr_model


models = lists_models() # get all available models 


model = load_ocr_model('model1') 


# load image and binarize it using the [kraken](https://kraken.re/main/index.html).ocr libray
image_path = "images/0086.jpg" 
image  = Image.open(image_path).convert('L')
 

text = model.ocr([image]) #  ['ἡμέραν ἐς τὸ συγκείμενον. τρίτος δέ ποτε ἐν']

``` 