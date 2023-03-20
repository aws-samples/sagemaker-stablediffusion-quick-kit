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
import shutil

from utils import quick_download_s3,get_bucket_and_key,untar

s3_bucket=os.environ.get("s3_bucket","sagemaker-us-east-1-1111111111")

def test_get_bucket_and_key():
    bucket,key=get_bucket_and_key(f"s3://{s3_bucket}/stablediffusion/asyncinvoke/images/bc7c889e-52eb-456f-a462-95a043bffef5.jpg")
    assert bucket==s3_bucket
    assert key=='stablediffusion/asyncinvoke/images/bc7c889e-52eb-456f-a462-95a043bffef5.jpg'
    

def test_quick_download_s3():
    #set correct s3 bucket test this 
    result=quick_download_s3(f's3://{s3_bucket}/stablediffusion/model/12340/vae/diffusion_pytorch_model.bin')
    assert result is not None
    result=quick_download_s3(f's3://{s3_bucket}/stablediffusion/model/12340')
    print(result)
    assert result is not None
    
    
def test_untar():
    output='/tmp/code'
    untar(f"/tmp/code.tgz",output)
    assert os.path.exists(output) is True
    shutil.rmtree(output)