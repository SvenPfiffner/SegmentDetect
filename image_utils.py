from PIL import Image
import torch
import numpy as np
from simple_lama_inpainting import SimpleLama
import cv2

def combine(image, preds_images, threshold=0.5):
    # Generate as many random colors as there are prompts
    colors = np.random.randint(0, 255, (len(preds_images), 3))
    colormask = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    for i, pred in enumerate(preds_images):
        colormask[pred == 1] = colors[i]

    # Overlay the original image with the predictions where the predictions are not black
    out = colormask.astype(np.float32) / 255
    image = np.array(image).astype(np.float32) / 255
    out_nonblack = out != [0, 0, 0]
    out_black = out == [0, 0, 0]
    out[out_nonblack] = out[out_nonblack] * 0.7 + image[out_nonblack] * 0.3
    out[out_black] = image[out_black]


    out = Image.fromarray((out * 255).astype("uint8"))
    return out

def build_masks(target_size, preds, threshold=0.5):
    preds_images = [torch.sigmoid(preds[i][0]).cpu().numpy() for i in range(preds.shape[0])]
    # Resize all predictions to the size of the original image
    preds_images = [np.array(Image.fromarray(pred).resize((target_size[0], target_size[1]))) for pred in preds_images]

    # Threshold each prediction
    masks = [(pred > threshold).astype("uint16") for pred in preds_images]
    return masks

def inpaint(image, masks, mask_dilation=100):
    simple_lama = SimpleLama()
    # Combine the masks into a single mask
    mask = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    for i, m in enumerate(masks):
        mask[m == 1] = np.ones(3)*255
    # Dilate the mask
    mask = cv2.dilate(mask, np.ones((mask_dilation, mask_dilation), np.uint8), iterations=1)

    mask = Image.fromarray(mask).convert("L")
    result = simple_lama(image, mask)
    
    return result