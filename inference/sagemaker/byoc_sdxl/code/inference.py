# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""inference module """
import os
import json
import io
import sys
import subprocess
import traceback
from PIL import Image

import requests
import boto3
import sagemaker
import torch

from typing import Optional
from torch import autocast
from diffusers import StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,DDIMScheduler

from diffusers.utils import load_image

#load utils and controle_net
from utils import download_model,write_imgage_to_s3
from control_net import ControlNetDectecProcessor,init_sdxl_control_net_model,init_sdxl_control_net_pipeline

from pydantic import BaseModel, Field

DEFAULT_MODEL="stabilityai/stable-diffusion-xl-base-1.0"

control_net_postfix=[
    "canny",
    "depth"
]


class Config(BaseModel):
    s3_bucket: str = os.environ.get("s3_bucket", "")
    custom_region: str = os.environ.get("custom_region", None)
    max_height: int = int(os.environ.get("max_height", 1024))
    max_width: int = int(os.environ.get("max_width", 1024))
    max_steps: int = int(os.environ.get("max_steps", 100))
    max_count: int = int(os.environ.get("max_count", 1))
    safety_checker_enable: bool = json.loads(os.environ.get("safety_checker_enable", "false"))
    model_name: str = os.environ.get("model_name", DEFAULT_MODEL)
    lora_name: str =os.environ.get("lora_name", "")
    lora_url:  str =os.environ.get("lora_url", "")
    watermarket: bool = json.loads(os.environ.get("watermarket","false"))
    watermarket_image: str=os.environ.get("watermarket_image", "sagemaker-logo-small.png")


class InferenceOpt(BaseModel):
    prompt: str = "a photo of an astronaut riding a horse on mars"
    negative_prompt: str = ""
    steps: int = 20
    sampler: Optional[str] = None
    height: int = 1024
    width: int = 1024
    count: int = Field(1,ge=1,le=1)
    seed: int = 1024
    input_image: Optional[str] = None
    init_image: Optional[str] = None
    control_net_model: str = ""
    control_net_detect: str = "true"
    control_net_enable:  str ="disable"
    sdxl_refiner: str = "disable"
    lora_name: str = ""
    lora_url: str = ""


config=Config()


processor=ControlNetDectecProcessor(size=(1024,1024))


result = subprocess.run(['df', '-kh'], stdout=subprocess.PIPE)
print("=========disk space=========")
print(result.stdout.decode())
print("============================")
#warm control net 
init_sdxl_control_net_model()
#warm SD XL model load from HF
pipe=StableDiffusionXLPipeline.from_pretrained(
    config.model_name,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir="/tmp/"
)
refiner=StableDiffusionXLImg2ImgPipeline.from_pretrained(
    config.model_name.replace('base','refiner'),
    torch_dtype=torch.float16,
    cache_dir="/tmp/"
)
#warm lora
if config.lora_name!=""  and config.lora_url!="" :
    lora_model_path=download_model(config.lora_url,model_name=f"{config.lora_name}.safetensors")
    print(f"download {lora_model_path}.safetensors")




    
def get_default_bucket():
    """
    get_default_bucket
    """
    try:
        sagemaker_session = sagemaker.Session() if config.custom_region is None else sagemaker.Session(
            boto3.Session(region_name=config.custom_region))
        bucket = sagemaker_session.default_bucket()
        return bucket
    except Exception as ex:
        if config.s3_bucket!="":
            return config.s3_bucket
        return None
            


# need add more sampler
samplers = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "eular": EulerDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "ddim": DDIMScheduler
}

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    #w, h = imgs[0].size
    w,h=1024,1024
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def predict_fn(opt:InferenceOpt):
    """
    Apply model to the incoming request
    """
    print("=================predict_fn=================")
    print('input_data: ', opt)
    prediction = []
    model_name = config.model_name
    watermark = config.watermarket
    watermark_image=config.watermarket_image
    

    try:
        bucket= get_default_bucket()
        if bucket is None:
            raise Exception("Need setup default bucket")
        
        init_image = opt.init_image
        input_image = opt.input_image
        print(f'{init_image=},{input_image=} ')

        
        if input_image is not None:
            response = requests.get(input_image, timeout=5)
            init_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            init_img = init_img.resize(
                (opt.width, opt.height))
           
                
        generator = torch.Generator(
            device='cuda').manual_seed(opt.seed)
        
        control_net_model_name=opt.control_net_model
        control_net_detect=opt.control_net_detect
        
        
        lora_model_path=""

        if opt.lora_name!="" and opt.lora_url!="":
            lora_model_path=download_model(model_name=opt.lora_name,model_url=opt.lora_url)

        #if model_name have stable-diffusion-xl prefix ,  process use SD XL and return 
        if "stable-diffusion-xl" in model_name and opt.control_net_enable=="disable":
            pipe=StableDiffusionXLPipeline.from_pretrained(
                                                        model_name,
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,
                                                        variant="fp16",
                                                        cache_dir="/tmp/"
                                                    )
            if lora_model_path!="":
                pipe.load_lora_weights(lora_model_path)
                pipe.to(torch_dtype=torch.float16)
                print(f"load lora model: {lora_model_path=}")
            pipe.to("cuda")

            
            images = pipe(opt.prompt,num_inference_steps=opt.steps).images
            if opt.sdxl_refiner=="enable":
                print(f'{opt.sdxl_refiner=}')
                refiner=StableDiffusionXLImg2ImgPipeline.from_pretrained(
                                            model_name.replace('base','refiner'),
                                            torch_dtype=torch.float16,
                                            cache_dir="/tmp/"
                                        )
                refiner.to("cuda")

                images = refiner(prompt=opt.prompt,num_inference_steps=opt.steps, image=images).images

            prediction=write_imgage_to_s3(images,watermark=(watermark==True),watermark_image=watermark_image,width=opt.width,height=opt.height)
            torch.cuda.empty_cache()
            return prediction
        
        if opt.control_net_enable=="enable":
            if control_net_detect=="true":
                print(f"detect_process {input_image}")
                control_net_input_image=processor.detect_process(control_net_model_name,input_image)
            else:
                control_net_input_image=load_image(input_image)
            
        with autocast("cuda"):
            if opt.control_net_enable=="enable":
                model_name = os.environ.get("model_name", DEFAULT_MODEL)
                pipe=init_sdxl_control_net_pipeline(model_name,opt.control_net_model)    
                pipe.enable_model_cpu_offload()
                
                images = pipe(opt.prompt, image=control_net_input_image, negative_prompt=opt.negative_prompt,
                               num_inference_steps=opt.steps, width=opt.width,height=opt.height,generator=generator).images
                grid_images=[]
                grid_images.insert(0,control_net_input_image)
                grid_images.insert(0,init_img)
                grid_images.extend(images)
                grid_image=image_grid(grid_images,1,len(grid_images))
                    
                if control_net_detect=="true":
                    images.append(control_net_input_image)
                images.append(grid_image)
                        
            else:
                images = model(opt.prompt, image=init_img, negative_prompt=opt.negative_prompt,
                                num_inference_steps=opt.steps, num_images_per_prompt=opt.count, generator=generator).images
            prediction=write_imgage_to_s3(images,watermark=(watermark==True),watermark_image=watermark_image,width=opt.width,height=opt.height)

            
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"=================Exception=================\n{ex}")

    print('prediction: ', prediction)
    return prediction


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    print(content_type)
    return json.dumps(
        {
            'result': prediction
        }
    )
