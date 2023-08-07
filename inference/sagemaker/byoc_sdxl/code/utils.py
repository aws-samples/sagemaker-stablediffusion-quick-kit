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

""" utils module"""
import io
import os
import uuid
import tarfile
import datetime
import subprocess

from PIL import Image
import boto3

def download_model(model_url,model_name="lora.safetensors"):
    """
    download model
    """
    try:
        if os.path.isfile(f"/tmp/{model_name}") is False:
            subprocess.run(["curl", "-L","-o", f"/tmp/{model_name}", model_url], check=True)
            print(f"download completed, /tmp/{model_name} ")       
        else:
            print(f"file already exits, /tmp/{model_name},skip download ")
        return f"/tmp/{model_name}"
    except subprocess.CalledProcessError as ex:
        print(f"download failed, {model_url} ex: {ex}")
        return None 

def quick_download_s3(s3_path, model_path="/tmp/models"):
    """
    :param fname: tar file name
    :param dirs: untar path
    :return: bool
    """
    base_name=os.path.basename(s3_path)
    _,ext_name=os.path.splitext(base_name)
    if s3_object_exists(s3_path) is False:
        return None
    if ext_name in [".ckpt",".pt",".safetensors",".bin"]:
        print("WARNING: this is single model file , may be need convert to diffuser")
    else:
        local_path= "/".join(s3_path.split("/")[-2:])
        model_path=f"{model_path}/{local_path}"
        print(f"need copy {s3_path} to {model_path}")
        os.makedirs(model_path,exist_ok=True)
        if s3_path.endswith("/") is False:
            s3_path=s3_path+"/"
        s3_path=s3_path+"*"
    command = f"/opt/conda/bin/s5cmd sync {s3_path} {model_path}"
    try:
        subprocess.run(command, shell=True)
        print(f"s3 download completed,cmd: {command}, model_path:{model_path}")
        return model_path
    except Exception as ex:
        print(ex)
        return None
        
def s3_object_exists(s3_path):
    """
    s3_object_exists
    """
    try:
        s3_client = boto3.client('s3')
        base_name=os.path.basename(s3_path)
        _,ext_name=os.path.splitext(base_name)
        bucket,key=get_bucket_and_key(s3_path)
        
        if ext_name!="":
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        if not key.endswith('/'):
            path = key+'/' 
            resp = s3_client.list_objects(Bucket=bucket, Prefix=path, Delimiter='/',MaxKeys=10)
            exists='Contents' in resp or 'CommonPrefixes' in resp
            return exists
    except Exception as ex:
        print(ex)
        return False

def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key

def untar(fname, dirs):
    """
    :param fname: tar file name
    :param dirs: untar path
    :return: bool
    """
    try:
        tar_file = tarfile.open(fname)
        tar_file.extractall(path = dirs)
        return True
    except Exception as ex:
        print(ex)
        return False

def write_imgage_to_s3(images,watermark=False,watermark_image="sagemaker-logo-small.png",width=512,height=512,output_s3uri=""):
    """
    write image to s3 bucket
    """
    s3_client = boto3.client('s3')
    s3_bucket = os.environ.get("s3_bucket", "")
    prediction = []
    
    
    default_output_s3uri = f's3://{s3_bucket}/stablediffusion/asyncinvoke/images/'
    if output_s3uri is None or output_s3uri=="":
        output_s3uri=default_output_s3uri
    
    if watermark:
        watermark_image_path=f"/opt/ml/model/{watermark_image}"
        if os.path.isfile(watermark_image_path) is False:
            watermark_image_path="/opt/program/sagemaker-logo-small.png"
        print(f"watermarket image path: {watermark_image_path}")
        crop_image = Image.open(watermark_image_path)
        size = (200, 39)
        crop_image.thumbnail(size)
        if crop_image.mode != "RGBA":
            crop_image = crop_image.convert("RGBA")
        layer = Image.new("RGBA",[width,height],(0,0,0,0))
        layer.paste(crop_image,(width-210, height-49))
    
    for image in images:
        bucket, key = get_bucket_and_key(output_s3uri)
        key = f'{key}{uuid.uuid4()}.jpg'
        buf = io.BytesIO()
        if watermark:
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
                "seed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        print('image: ', f's3://{bucket}/{key}')
        prediction.append(f's3://{bucket}/{key}')
    return prediction
