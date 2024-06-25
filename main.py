from PIL import Image
import requests
from segmenter import Segmenter

from image_combine import combine


url = "https://unsplash.com/photos/8Nc_oQsc2qQ/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjcxMjAwNzI0&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)
image_size = image.size

prompts = ["cutlery", "pancakes", "blueberries", "orange juice"]

segmenter = Segmenter()
preds = segmenter.segment(image, prompts)

out = combine(image, preds)

out.show()

exit(0)
print(image.size)
print(preds.shape)
import matplotlib.pyplot as plt

_, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.axis('off')
combined_headmap = torch.sigmoid(torch.cat(preds, dim=0).max(dim=0)[0])
ax.imshow(combined_headmap)
ax.text(0, -15, ', '.join(prompts))
plt.savefig("test.png")

