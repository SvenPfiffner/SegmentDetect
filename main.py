from PIL import Image
import requests
import segmentdetect


url = "https://unsplash.com/photos/8Nc_oQsc2qQ/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjcxMjAwNzI0&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
image_size = image.size


prompts = ["cutlery", "pancakes", "blueberries", "orange juice"]

segmenter = segmentdetect.Segmenter()
preds = segmenter.get_segmentation_masks(image, prompts)
