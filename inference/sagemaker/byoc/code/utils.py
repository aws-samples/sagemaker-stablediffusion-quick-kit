import os 
import traceback
import tarfile

import subprocess

import boto3


def quick_download_s3(s3_path, model_path="/tmp/models"):
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
    subprocess.run(command, shell=True)
    print(f"s3 download completed,cmd: {command}, model_path:{model_path}")
    return model_path
   

def s3_object_exists(s3_path):
    """
    s3_object_exists
    """
    try:
        s3 = boto3.client('s3')
        base_name=os.path.basename(s3_path)
        _,ext_name=os.path.splitext(base_name)
        bucket,key=get_bucket_and_key(s3_path)
        
        if ext_name!="":
            s3.head_object(Bucket=bucket, Key=key)
            return True
        else:
            if not key.endswith('/'):
                path = key+'/' 
                resp = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter='/',MaxKeys=10)
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
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False