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

import os
import json
import uuid
import io
import sys
import tarfile
import traceback

from PIL import Image

import requests
import boto3
import sagemaker
import torch

from PIL import Image
from transformers import pipeline as depth_pipeline

from torch import autocast
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline
from diffusers import AltDiffusionPipeline, AltDiffusionImg2ImgPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,DDIMScheduler

from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector,MLSDdetector,HEDdetector,HEDdetector

from diffusers.utils import load_image


import cv2
import numpy as np

#load utils and controle_net
from utils import quick_download_s3,get_bucket_and_key,untar
from control_net import ControlNetDectecProcessor,init_control_net_pipeline,init_control_net_model



s3_client = boto3.client('s3')


max_height = os.environ.get("max_height", 768)
max_width = os.environ.get("max_width", 768)
max_steps = os.environ.get("max_steps", 100)
max_count = os.environ.get("max_count", 4)
s3_bucket = os.environ.get("s3_bucket", "")
watermarket=os.environ.get("watermarket", True)
watermarket_image=os.environ.get("watermarket_image", "sagemaker-logo-small.png")
custom_region = os.environ.get("custom_region", None)
safety_checker_enable = json.loads(os.environ.get("safety_checker_enable", "false"))
control_net_enable=os.environ.get("control_net_enable", "enable")
deepspeed_enable=os.environ.get("deepspeed", False)

DEFAULT_MODEL="runwayml/stable-diffusion-v1-5"


processor=ControlNetDectecProcessor()
init_control_net_model()



#control_net
control_net_prefix="lllyasviel/sd-controlnet"
control_net_postfix=[
    "canny",
    "depth",
    "hed",
    "mlsd",
    "openpose",
    "scribble"
]



controle_net_cache={}

def check_chontrole_net(model_list):
    model_list=model_list.split(",")
    valid_model=[]
    for model in model_list:
        if model in control_net_postfix:
            valid_model.append(f"{model}")
    print(f"valid_control_net model: {valid_model} ")
    return valid_model
    



def canny_image_detector(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    return canny_image

    
def get_default_bucket():
    try:
        sagemaker_session = sagemaker.Session() if custom_region is None else sagemaker.Session(
            boto3.Session(region_name=custom_region))
        bucket = sagemaker_session.default_bucket()
        return bucket
    except Exception as ex:
        if s3_bucket!="":
            return s3_bucket
        else:
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

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid




def init_pipeline(model_name: str,model_args=None):
    """
    help load model from s3
    """
    print(f"=================init_pipeline:{model_name}=================")
    
    if control_net_enable:
        model_name=DEFAULT_MODEL if "s3" in model_name else model_name
        controlnet_model = ControlNetModel.from_pretrained(f"{control_net_prefix}-canny", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet_model,torch_dtype=torch.float16
        )
        print(f"load {model_name} with controle net")
        return pipe

                
    model_path=model_name
    base_name=os.path.basename(model_name)
    try:
        if model_name.startswith("s3://"):
            if base_name=="model.tar.gz":
                local_path= "/".join(model_name.split("/")[-2:-1])
                model_path=f"/tmp/{local_path}"
                print(f"need copy {model_name} to {model_path}")
                os.makedirs(model_path)
                fs.get(model_name,model_path+"/", recursive=True)
                untar(f"/tmp/{local_path}/model.tar.gz",model_path)
                os.remove(f"/tmp/{local_path}/model.tar.gz")
                print("download and untar  completed")
            else:
                local_path= "/".join(model_name.split("/")[-2:])
                model_path=f"/tmp/{local_path}"
                print(f"need copy {model_name} to {model_path}")
                os.makedirs(model_path)
                fs.get(model_name,model_path, recursive=True)
                print("download completed")

        print(f"pretrained model_path: {model_path}")
        if model_args is not None:
            return StableDiffusionPipeline.from_pretrained(
                 model_path, **model_args)
        return StableDiffusionPipeline.from_pretrained(model_path)
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"=================Exception================={ex}")
        return None

    
model_name = os.environ.get("model_name", DEFAULT_MODEL)
model_args = json.loads(os.environ['model_args']) if (
        'model_args' in os.environ) else None
#warm model load 
init_pipeline(model_name,model_args)


def model_fn(model_dir):
    """
    Load the model for inference,load model from os.environ['model_name'],diffult use runwayml/stable-diffusion-v1-5
    
    """
    print("=================model_fn=================")
    print(f"model_dir: {model_dir}")
    model_name = os.environ.get("model_name", DEFAULT_MODEL)
    model_args = json.loads(os.environ['model_args']) if (
        'model_args' in os.environ) else None
    task = os.environ['task'] if ('task' in os.environ) else "text-to-image"
    print(
        f'model_name: {model_name},  model_args: {model_args}, task: {task} ')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

   
    model = init_pipeline(model_name,model_args)
    
    if safety_checker_enable is False :
        #model.safety_checker = lambda images, clip_input: (images, False)
        model.safety_checker=None
    if deepspeed_enable:
        try:
            print("begin load deepspeed....")
            model=deepspeed.init_inference(
                model=getattr(model,"model", model),      # Transformers models
                mp_size=1,        # Number of GPU
                dtype=torch.float16, # dtype of the weights (fp16)
                replace_method="auto", # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=False, # replace the model with the kernel injector
            )
            print("model accelarate with deepspeed!")
        except Exception as e:
            print("deepspeed accelarate excpetion!")
            print(e)
        
    
    model = model.to("cuda")
    model.enable_attention_slicing()

    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    print(f"=================input_fn=================\n{request_content_type}\n{request_body}")
    input_data = json.loads(request_body)
    return prepare_opt(input_data)


def clamp_input(input_data, minn, maxn):
    """
    clamp_input check input 
    """
    return max(min(maxn, input_data), minn)


def prepare_opt(input_data):
    """
    Prepare inference input parameter
    """
    opt = {}
    opt["prompt"] = input_data.get(
        "prompt", "a photo of an astronaut riding a horse on mars")
    opt["negative_prompt"] = input_data.get("negative_prompt", "")
    opt["steps"] = clamp_input(input_data.get(
        "steps", 20), minn=20, maxn=max_steps)
    opt["sampler"] = input_data.get("sampler", None)
    opt["height"] = clamp_input(input_data.get(
        "height", 512), minn=64, maxn=max_height)
    opt["width"] = clamp_input(input_data.get(
        "width", 512), minn=64, maxn=max_width)
    opt["count"] = clamp_input(input_data.get(
        "count", 1), minn=1, maxn=max_count)
    opt["seed"] = input_data.get("seed", 1024)
    opt["input_image"] = input_data.get("input_image", None)
    opt["control_net_model"] = input_data.get("control_net_model","")
    opt["control_net_detect"] = input_data.get("control_net_detect","true")
    
    if  opt["control_net_model"] not in control_net_postfix:
        opt["control_net_model"]=""
    

    if opt["sampler"] is not None:
        opt["sampler"] = samplers[opt["sampler"]
                                  ] if opt["sampler"] in samplers else samplers["euler_a"]

    print(f"=================prepare_opt=================\n{opt}")
    return opt


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    print("=================predict_fn=================")
    print('input_data: ', input_data)
    prediction = []

    try:


        bucket= get_default_bucket()
    
        if bucket is None:
            raise Exception("Need setup default bucket")
        default_output_s3uri = f's3://{bucket}/stablediffusion/asyncinvoke/images/'
        output_s3uri = input_data['output_s3uri'] if 'output_s3uri' in input_data else default_output_s3uri
        infer_args = input_data['infer_args'] if (
            'infer_args' in input_data) else None
        print('infer_args: ', infer_args)
        init_image = infer_args['init_image'] if infer_args is not None and 'init_image' in infer_args else None
        input_image = input_data['input_image']
        print('init_image: ', init_image)
        print('input_image: ', input_image)

        # load different Pipeline for txt2img , img2img
        # referen doc: https://huggingface.co/docs/diffusers/api/diffusion_pipeline#diffusers.DiffusionPipeline.components
        #   text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        #   img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        #   inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        #  use StableDiffusionImg2ImgPipeline for input_image        
        if input_image is not None:
            response = requests.get(input_image, timeout=5)
            init_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            init_img = init_img.resize(
                (input_data["width"], input_data["height"]))
            if control_net_enable is False:
                model = StableDiffusionImg2ImgPipeline(**model.components)  # need use Img2ImgPipeline
                
                
        generator = torch.Generator(
            device='cuda').manual_seed(input_data["seed"])
        
        control_net_model_name=input_data.get("control_net_model")
        control_net_detect=input_data.get("control_net_detect")
        if control_net_enable:
            if control_net_detect=="true":
                print(f"detect_process {input_image}")
                control_net_input_image=processor.detect_process(control_net_model_name,input_image)
            else:
                control_net_input_image=load_image(input_image)
        
        with autocast("cuda"):
            if model != None:
                model.scheduler = input_data["sampler"].from_config(
                    model.scheduler.config)
            if input_image is None:
                images = model(input_data["prompt"], input_data["height"], input_data["width"], negative_prompt=input_data["negative_prompt"],
                               num_inference_steps=input_data["steps"], num_images_per_prompt=input_data["count"], generator=generator).images
            else:
                if control_net_enable:
                    model_name = os.environ.get("model_name", DEFAULT_MODEL)
                    pipe=init_control_net_pipeline(model_name,input_data["control_net_model"])    
                    pipe.enable_model_cpu_offload()
                    images = pipe(input_data["prompt"], image=control_net_input_image, negative_prompt=input_data["negative_prompt"],
                               num_inference_steps=input_data["steps"], generator=generator).images
                    grid_images=[]
                    grid_images.insert(0,control_net_input_image)
                    grid_images.insert(0,init_img)
                    grid_images.extend(images)
                    grid_image=image_grid(grid_images,1,len(grid_images))
                    
                    if control_net_detect=="true":
                        images.append(control_net_input_image)
                    images.append(grid_image)
                        
                else:
                    images = model(input_data["prompt"], image=init_img, negative_prompt=input_data["negative_prompt"],
                               num_inference_steps=input_data["steps"], num_images_per_prompt=input_data["count"], generator=generator).images
            # image watermark
            if watermarket:
                watermarket_image_path=f"/opt/ml/model/{watermarket_image}"
                if os.path.isfile(watermarket_image_path) is False:
                    watermarket_image_path="/opt/program/sagemaker-logo-small.png"
                print(f"watermarket image path: {watermarket_image_path}")
                crop_image = Image.open(watermarket_image_path)
                size = (200, 39)
                crop_image.thumbnail(size)
                if crop_image.mode != "RGBA":
                    crop_image = crop_image.convert("RGBA")
                layer = Image.new("RGBA",[input_data["width"],input_data["height"]],(0,0,0,0))
                layer.paste(crop_image,(input_data["width"]-210, input_data["height"]-49))
            
            for image in images:
                bucket, key = get_bucket_and_key(output_s3uri)
                key = f'{key}{uuid.uuid4()}.jpg'
                buf = io.BytesIO()
                if watermarket:
                    out = Image.composite(layer,image,layer)
                    out.save(buf, format='JPEG')
                else:
                    image.save(buf, format='JPEG')
                
                s3_client.put_object(
                    Body=buf.getvalue(),
                    Bucket=bucket,
                    Key=key,
                    ContentType='image/jpeg',
                    Metadata={
                        # #s3 metadata only support ascii
                        "seed": str(input_data["seed"])
                    }
                )
                print('image: ', f's3://{bucket}/{key}')
                prediction.append(f's3://{bucket}/{key}')
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"=================Exception================={ex}")

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
