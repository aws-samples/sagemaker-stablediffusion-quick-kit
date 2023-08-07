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


import requests
import boto3
import sagemaker
import torch


from PIL import Image
from transformers import pipeline as depth_pipeline

from torch import autocast
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionXLPipeline,StableDiffusionXLImg2ImgPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,DDIMScheduler


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


from diffusers.utils import load_image



#load utils and controle_net
from utils import download_model,write_imgage_to_s3
from control_net import ControlNetDectecProcessor,init_control_net_pipeline,init_control_net_pipeline_v1_1,init_control_net_model,init_control_net_model_v1_1


DEFAULT_MODEL="runwayml/stable-diffusion-v1-5"
DEFAULT_SD_XL_MODEL="stabilityai/stable-diffusion-xl-base-1.0"


s3_bucket = os.environ.get("s3_bucket", "")
custom_region = os.environ.get("custom_region", None)


#inference check
max_height = os.environ.get("max_height", 768)
max_width = os.environ.get("max_width", 768)
max_steps = os.environ.get("max_steps", 100)
max_count = os.environ.get("max_count", 4)
safety_checker_enable = json.loads(os.environ.get("safety_checker_enable", "false"))


model_name = os.environ.get("model_name", DEFAULT_MODEL)
model_args = json.loads(os.environ['model_args']) if (
        'model_args' in os.environ) else None

#control_net config , notice control_net v1.1 not ready for SDXL
control_net_enable=os.environ.get("control_net_enable", "disable")
control_net_version=os.environ.get("control_net_version","v1.1")
CONTROL_NET_PREFIX="lllyasviel/sd-controlnet"
control_net_postfix=[
    "canny",
    "depth",
    "hed",
    "mlsd",
    "openpose",
    "scribble"
]

#lora config
lora_name=os.environ.get("lora_name", None)
lora_url=os.environ.get("lora_url", None)

#watermarket
watermarket = json.loads(os.environ.get("watermarket","false"))
watermarket_image=os.environ.get("watermarket_image", "sagemaker-logo-small.png")

processor=ControlNetDectecProcessor()
lora_model_path = None

result = subprocess.run(['df', '-kh'], stdout=subprocess.PIPE)
print("=========disk=========")
print(result)

#warm control net 
if control_net_enable=="enable":
    if control_net_version=="v1.1":
        init_control_net_model_v1_1()
    else:
        init_control_net_model()
else:
    #warm SD XL model load from HF
    if "stable-diffusion-xl" in model_name:
         pipe=StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                cache_dir="/tmp/"
            )
         refiner=StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_name.replace('base','refiner'),
                torch_dtype=torch.float16,
                cache_dir="/tmp/"
            )

if lora_name  and lora_url :
    lora_model_path=download_model(lora_url,model_name=f"{lora_name}.safetensors")
    print(f"download {lora_model_path}.safetensors")




    
def get_default_bucket():
    """
    get_default_bucket
    """
    try:
        sagemaker_session = sagemaker.Session() if custom_region is None else sagemaker.Session(
            boto3.Session(region_name=custom_region))
        bucket = sagemaker_session.default_bucket()
        return bucket
    except Exception as ex:
        if s3_bucket!="":
            return s3_bucket
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
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def init_pipeline(model_name: str,model_args=None):
    """
    help load model from s3
    """
    print(f"=================init_pipeline:{model_name}=================")
    try:
        if control_net_enable=="enable":
            model_name=DEFAULT_MODEL if "s3" in model_name else model_name
            controlnet_model = ControlNetModel.from_pretrained(f"{CONTROL_NET_PREFIX}-canny", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_name, controlnet=controlnet_model,torch_dtype=torch.float16
            )
            print(f"load {model_name} with controle net")
            return pipe
        else:
            #if use SD XL model
            if "stable-diffusion-xl" in model_name:
                pipe=StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir="/tmp/"
                )
            else:
                pipe=StableDiffusionPipeline.from_pretrained(model_name)
            return pipe
                
    
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"=================Exception================={ex}")
        return None

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
        model.safety_checker=None

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
    opt["SDXL_REFINER"]=input_data.get("SDXL_REFINER","disable")
    opt["lora_name"]=input_data.get("lora_name","")
    opt["lora_url"]=input_data.get("lora_url","")
    
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
    model_name = os.environ.get("model_name", DEFAULT_MODEL)
    watermark = json.loads(os.environ.get("watermark","false"))
    watermark_image=os.environ.get("watermark_image", "sagemaker-logo-small.png")

    try:
        bucket= get_default_bucket()
        if bucket is None:
            raise Exception("Need setup default bucket")
        infer_args = input_data['infer_args'] if (
            'infer_args' in input_data) else None
        print('infer_args: ', infer_args)
        init_image = infer_args['init_image'] if infer_args is not None and 'init_image' in infer_args else None
        input_image = input_data['input_image']
        print('init_image: ', init_image)
        print('input_image: ', input_image,control_net_enable)

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
        
        
        lora_model_path=""

        if input_data["lora_name"]!="" and input_data["lora_url"]!="":
            lora_model_path=download_model(model_name=input_data["lora_name"],model_url=input_data["lora_url"])

        #if model_name have stable-diffusion-xl prefix ,  process use SD XL and return 
        if "stable-diffusion-xl" in model_name:
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

            
            images = pipe(input_data["prompt"],num_inference_steps=input_data["steps"]).images
            if input_data["SDXL_REFINER"]=="enable":
                print(f'{input_data["SDXL_REFINER"]=}')
                refiner=StableDiffusionXLImg2ImgPipeline.from_pretrained(
                                            model_name.replace('base','refiner'),
                                            torch_dtype=torch.float16,
                                            cache_dir="/tmp/"
                                        )
                refiner.to("cuda")

                images = refiner(prompt=input_data["prompt"],num_inference_steps=input_data["steps"], image=images).images

            prediction=write_imgage_to_s3(images,watermark=(watermark==True),watermark_image=watermark_image,width=input_data["width"],height=input_data["height"])
            torch.cuda.empty_cache()
            return prediction
        
        if control_net_enable=="enable":
            if control_net_detect=="true":
                print(f"detect_process {input_image}")
                control_net_input_image=processor.detect_process(control_net_model_name,input_image)
            else:
                control_net_input_image=load_image(input_image)
            
        with autocast("cuda"):
            if control_net_enable=="enable":
                model_name = os.environ.get("model_name", DEFAULT_MODEL)
                if control_net_version=="v1.1":
                    pipe=init_control_net_pipeline_v1_1(model_name,input_data["control_net_model"])    
                    pipe.enable_model_cpu_offload()
                else:
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
            prediction=write_imgage_to_s3(images,watermark=(watermark==True),watermark_image=watermark_image,width=input_data["width"],height=input_data["height"])

            
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
