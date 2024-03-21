import sys

sys.path.append("./ImageBind/")

import imagebind
import torch
import time
from diffusers import StableUnCLIPImg2ImgPipeline


class Generator(object):
    def __init__(self, num_inference_steps=30):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ib_model = imagebind.imagebind_model.imagebind_huge(pretrained=True)
        self.diffusion_model = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float32
        )
        self.ib_model.eval()
        self.ib_model.to(self.device)
        self.diffusion_model.to(self.device)
        self.num_inference_steps = num_inference_steps

    def text2image(self, text, filename):
        with torch.no_grad():
            ib_input = {
                imagebind.imagebind_model.ModalityType.TEXT: imagebind.data.load_and_transform_text(
                    [text], self.device
                ),
            }
            text_embeddings = self.ib_model(ib_input)
            decoder_input = text_embeddings[imagebind.imagebind_model.ModalityType.TEXT]
            images = self.diffusion_model(
                image_embeds=decoder_input, num_inference_steps=self.num_inference_steps
            ).images[0]
            images.save(f"output/{filename}.png")

    def image2image(self, input_filename, filename):
        with torch.no_grad():
            ib_input = {
                imagebind.imagebind_model.ModalityType.VISION: imagebind.data.load_and_transform_vision_data(
                    [input_filename], self.device
                ),
            }
            embeddings = self.ib_model(ib_input)
            decoder_input = embeddings[imagebind.imagebind_model.ModalityType.VISION]
            images = self.diffusion_model(
                image_embeds=decoder_input, num_inference_steps=self.num_inference_steps
            ).images[0]
            images.save(f"output/{filename}.png")

    def imagetext2image(self, text, input_filename, filename):
        with torch.no_grad():
            img_w = 0.4
            ib_input = {
                imagebind.imagebind_model.ModalityType.VISION: imagebind.data.load_and_transform_vision_data(
                    [input_filename], self.device
                ),
                imagebind.imagebind_model.ModalityType.TEXT: imagebind.data.load_and_transform_text(
                    [text], self.device
                ),
            }
            embeddings = self.ib_model(ib_input)
            decoder_input = (
                img_w * embeddings[imagebind.imagebind_model.ModalityType.VISION]
                + (1 - img_w) * embeddings[imagebind.imagebind_model.ModalityType.TEXT]
            )
            images = self.diffusion_model(
                image_embeds=decoder_input, num_inference_steps=self.num_inference_steps
            ).images[0]
            images.save(f"output/{filename}.png")
    
    def imageimage2image(self, input_filename1, input_filename2, filename):
        with torch.no_grad():
            img_w = 0.5
            ib_input = {
                imagebind.imagebind_model.ModalityType.VISION: imagebind.data.load_and_transform_vision_data(
                    [input_filename1, input_filename2], self.device
                )
            }
            embeddings = self.ib_model(ib_input)
            decoder_input = (
                img_w * embeddings[imagebind.imagebind_model.ModalityType.VISION][0]
                + (1 - img_w) * embeddings[imagebind.imagebind_model.ModalityType.VISION][1]
            ).view(1, -1)
            images = self.diffusion_model(
                image_embeds=decoder_input, num_inference_steps=self.num_inference_steps
            ).images[0]
            images.save(f"output/{filename}.png")
    
    def audiotext2image(self, text, audio_filename, filename):
        with torch.no_grad():
            audio_w = 0.35
            ib_input = {
                imagebind.imagebind_model.ModalityType.AUDIO: imagebind.data.load_and_transform_audio_data(
                    [audio_filename], self.device
                ),
                imagebind.imagebind_model.ModalityType.TEXT: imagebind.data.load_and_transform_text(
                    [text], self.device
                ),
            }
            embeddings = self.ib_model(ib_input)
            decoder_input = (
                audio_w * embeddings[imagebind.imagebind_model.ModalityType.AUDIO]
                + (1 - audio_w) * embeddings[imagebind.imagebind_model.ModalityType.TEXT]
            )
            images = self.diffusion_model(
                image_embeds=decoder_input, num_inference_steps=self.num_inference_steps
            ).images[0]
            images.save(f"output/{filename}.png")

    def audioimage2image(self, img_filename, audio_filename, filename):
        with torch.no_grad():
            audio_w = 0.2
            ib_input = {
                imagebind.imagebind_model.ModalityType.AUDIO: imagebind.data.load_and_transform_audio_data(
                    [audio_filename], self.device
                ),
                imagebind.imagebind_model.ModalityType.VISION: imagebind.data.load_and_transform_vision_data(
                    [img_filename], self.device
                ),
            }
            embeddings = self.ib_model(ib_input)
            decoder_input = (
                audio_w * embeddings[imagebind.imagebind_model.ModalityType.AUDIO]
                + (1 - audio_w) * embeddings[imagebind.imagebind_model.ModalityType.VISION]
            )
            images = self.diffusion_model(
                image_embeds=decoder_input, num_inference_steps=self.num_inference_steps
            ).images[0]
            images.save(f"output/{filename}.png")
