from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

class Segmenter:

    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


    def segment(self, image, prompts):
        inputs = self.processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
        # predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = outputs.logits.unsqueeze(1)

        return preds