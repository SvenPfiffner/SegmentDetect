from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np
from PIL import Image

class Segmenter:

    def __init__(self):
        """
        Initializes a Segmenter object.

        This method initializes the Segmenter object by loading the CLIPSegProcessor and CLIPSegForImageSegmentation models.
        Note: On the first run, this method will download the models from the Hugging Face model hub, which may take some time.
        """
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


    def segment(self, image, prompts):
        """
        Segments the given image based on the provided prompts and returns the segmented image as a tensor.

        Args:
            image (PIL.Image.Image) or (np.ndarray): The image to segment.
            prompts (List[str]): A list of prompts to guide the segmentation process.

        Returns:
            torch.Tensor: The segmented image as a tensor.

        """

        inputs = self.processor(text=prompts, images=[image] * len(prompts), return_tensors="pt")
        # predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = outputs.logits.unsqueeze(1)

        return preds
    
    def get_segmentation_masks(self, image, prompts, threshold=0.5):
        """
        Segments the given image based on the provided prompts and returns the segmented image as a collection of binary masks.

        Args:
            image (PIL.Image.Image) or (np.ndarray): The image to segment.
            prompts (list): A list of prompts to guide the segmentation process.
            threshold (float, optional): The threshold value for binarizing the predictions. Defaults to 0.5.

        Returns:
            dict: A dictionary containing the binary masks for each prompt as numpy arrays.
        """

        preds = self.segment(image, prompts)
        assert len(preds) == len(prompts), "The number of predictions must match the number of prompts."

        preds_images = [torch.sigmoid(preds[i][0]).cpu().numpy() for i in range(preds.shape[0])]
        # Resize all predictions to the size of the original image
        preds_images = [np.array(Image.fromarray(pred).resize(image.size)) for pred in preds_images]

        # Threshold each prediction and store as a binary mask
        masks = {prompts[i]: (pred > threshold).astype("uint16") for i, pred in enumerate(preds_images)}
        return masks
    
    def overlay(image, masks):
        """
        Overlay the original image with the predicted masks.

        Args:
            image (PIL.Image.Image): The original image.
            masks (dict): A dictionary of predicted masks.

        Returns:
            PIL.Image.Image: The overlayed image.

        """
        # Generate as many random colors as there are prompts
        colors = np.random.randint(0, 255, (len(masks.values()), 3))
        colormask = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
        for i, pred in enumerate(masks.values()):
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