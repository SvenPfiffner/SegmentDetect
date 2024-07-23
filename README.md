# SegmentDetect

A lightweight python package to perform zero-shot image segmentation via CLIP.

This repository implements promptable image segmentation in the zero-shot setting as discussed in
- [Image Segmentation Using Text and Image Prompts, 2022, Lüddecke & Ecker](https://arxiv.org/pdf/2112.10003)

by wrapping the [```CLIPSegProcessor```](https://huggingface.co/docs/transformers/en/model_doc/clipseg) of the ```transformer``` package in a lightweight segmenter class

<div align="center">

<img src="https://github.com/SvenPfiffner/SegmentDetect/blob/main/demo.jpg" width="350">

</div>

## How to use
**UI**
The package allows demoing the produced masks via a streamlit webui.
*TODO: Add package command*

**In code**
To perform segmentation, first create a new ```Segmenter``` object. *Note: on first use, the constructor will load the required models from Huggingface. So the first run might take some additional time*

```python
import segmentdetect

segmenter = segmentdetect.Segmenter()
```

To produce a segmentation tensor, call the ```segment()``` method with an image and a list of prompts

```python
from PIL import Image

img = Image.open(...)
prompts = ["Human", "Car", ...]

segmentation = segmenter.segment(img, prompts)
```

Alternatively, you can use the ```get_segmentation_masks()``` wrapper method that returns the segmentation as a dictionary containing prompts as keys and corresponding binary masks as values (predictions are thresholded by the optional threshold value)

```python
from PIL import Image

img = Image.open(...)
prompts = ["Human", "Car", ...]

segmentation = segmenter.get_segmentation_masks(img,
                prompts, threshold=0.8)
```

For debugging and prototyping, it might be useful to overlay the predicted masks over the original image. To do so, pass your original image and the retrieved binary masks to the static ```overlay()``` function

```python
from PIL import Image

img = Image.open(...)
prompts = ["Human", "Car", ...]

segmentation = segmenter.get_segmentation_masks(img,
                prompts, threshold=0.8)

debug_img = egmentdetect.Segmenter.overlay(img, segmentation)
```