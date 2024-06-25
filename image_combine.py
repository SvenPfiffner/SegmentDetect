from PIL import Image
import torch
import numpy as np

def combine(image, preds, threshold=0.5):
    preds_images = [torch.sigmoid(preds[i][0]).cpu().numpy() for i in range(preds.shape[0])]
    # Resize all predictions to the size of the original image
    preds_images = [np.array(Image.fromarray(pred).resize((image.size[0], image.size[1]))) for pred in preds_images]

    # Generate as many random colors as there are prompts
    colors = np.random.randint(0, 255, (len(preds_images), 3))

    # Threshold each prediction and fill the area with a random color

    out = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    print(out.shape)
    for i, pred in enumerate(preds_images):
        pred = (pred > threshold).astype(np.uint8)
        out[pred == 1] = colors[i]

    # Overlay the original image with the predictions where the predictions are not black
    out = out.astype(np.float32) / 255
    image = np.array(image).astype(np.float32) / 255
    out_nonblack = out != [0, 0, 0]
    out_black = out == [0, 0, 0]
    out[out_nonblack] = out[out_nonblack] * 0.7 + image[out_nonblack] * 0.3
    out[out_black] = image[out_black]


    out = Image.fromarray((out * 255).astype("uint8"))
    return out