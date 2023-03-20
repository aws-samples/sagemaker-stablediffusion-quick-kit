import os
import json
import uuid
import io
import sys
import tarfile
import traceback

from PIL import Image

import torch
from transformers import pipeline as depth_pipeline
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector,MLSDdetector,HEDdetector,HEDdetector

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler



import cv2
import numpy as np


control_net_postfix=[
                                    "canny",
                                    "depth",
                                    "hed",
                                    "mlsd",
                                    "openpose",
                                    "scribble"
                                ]



class ControlNetDectecProcessor:
    def __init__(self):
        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        self.depth= depth_pipeline('depth-estimation')
        
    
    def detect_process(self,model_name,image_url):
        if model_name not in control_net_postfix:
            return None
        func = getattr(ControlNetDectecProcessor, f'get_{model_name}_image')
        return func(self,image_url)
    
        
    def get_openpose_image(self,image_url):
        image = load_image(image_url)
        pose_image = self.openpose(image)
        return pose_image

    def get_mlsd_image(self,image_url):
        image = load_image(image_url)
        mlsd_image = self.mlsd(image)
        return mlsd_image

    def get_hed_image(self,image_url):
        image = load_image(image_url)
        hed_image = self.hed(image)
        return hed_image

    def get_scribble_image(self,image_url):
        image = load_image(image_url)
        scribble_image = self.hed(image,scribble=True)
        return scribble_image

    def get_depth_image(self,image_url):
        image = load_image(image_url)
        depth_image = self.depth(image)['depth']
        depth_image = np.array(depth_image)
        depth_image = depth_image[:, :, None]
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
        depth_image = Image.fromarray(depth_image)
        return depth_image

    def get_canny_image(self,image_url):
        image = load_image(image_url)
        image = np.array(image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image


def init_control_net_model():
    print(f"init_control_net_model:{control_net_postfix} begain")
    for model in control_net_postfix:
        controlnet = ControlNetModel.from_pretrained(
                                    f"lllyasviel/sd-controlnet-{model}", torch_dtype=torch.float16
                            )
    print(f"init_control_net_model:{control_net_postfix} completed")
    
def init_control_net_pipeline(base_model,control_net_model):
    if control_net_model not in control_net_postfix:
            return None
    controlnet = ControlNetModel.from_pretrained(
                                    f"lllyasviel/sd-controlnet-{control_net_model}", torch_dtype=torch.float16
                            )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                                base_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
                                )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


        
    
    return pipe
        
    
