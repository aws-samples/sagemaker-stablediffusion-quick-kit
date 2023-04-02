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

import json
import uuid
import os
import urllib
import traceback
import sys
import hashlib
import boto3
import urllib3
import base64

from json import JSONEncoder

from botocore.exceptions import ClientError

CURRENT_REGION= boto3.session.Session().region_name


from botocore.vendored import requests
from datetime import datetime

SM_REGION=os.environ.get("SM_REGION") if os.environ.get("SM_REGION")!="" else CURRENT_REGION
SM_ENDPOINT=os.environ.get("SM_ENDPOINT",None) #SM_ENDPORT NAME
S3_BUCKET=os.environ.get("S3_BUCKET","")
S3_PREFIX=os.environ.get("S3_PREFIX","stablediffusion/asyncinvoke")
CDN_BASE=os.environ.get("CDN_BASE","") #cloudfront base uri
DDB_TABLE=os.environ.get("DDB_TABLE","") #dynamodb table name
GALLERY_ADMIN_TOKEN=os.environ.get("GALLERY_ADMIN_TOKEN","") #gallery admin token 


if CDN_BASE.startswith("https") is False:
    CDN_BASE=f'https://{CDN_BASE}'

print(f"CURRENT_REGION |{CURRENT_REGION}|")
print(f"SM_REGION |{SM_REGION}|")
print(f"SM_ENDPOINT |{SM_ENDPOINT}|")

print(f"S3_BUCKET |{S3_BUCKET}|")
print(f"S3_PREFIX |{S3_PREFIX}|")
print(f"CDN_BASE |{CDN_BASE}|")


sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=SM_REGION)
s3_client = boto3.client("s3")

session = boto3.Session()
s3 = session.resource('s3')

class APIconfig:

    def __init__(self, item,include_attr=True):
        if include_attr:
            self.sm_endpoint = item.get('SM_ENDPOINT').get('S')
            self.label = item.get('LABEL').get('S'),
            self.hit = item.get('HIT',{}).get('S','')
            self.publish = item.get('PUBLISH',{}).get('BOOL',False)
        else:
            self.sm_endpoint = item.get('SM_ENDPOINT')
            self.label = item.get('LABEL') 
            self.hit = item.get('HIT','')
            self.publish = item.get('PUBLISH',False)


    def __repr__(self):
        return f"APIconfig<{self.label} -- {self.sm_endpoint}>"
        

class APIConfigEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__
            
def download_and_upload(image_url):
    http = urllib3.PoolManager()
    response = http.request('GET', image_url)
    image_data = response.data
    s3 = boto3.client('s3')
    key = os.path.basename(image_url) # key is the name of file on your bucket
    time = datetime.now()
    time_prefix=time.strftime("%Y/%m/%d")
    full_key = f'gallery/{time_prefix}/{key}'
    s3.put_object(Body=image_data, Bucket=S3_BUCKET, Key=full_key)
    print(f'download_and_upload: {full_key}')
    if CDN_BASE=="":
        return (f's3://{S3_BUCKET}/{full_key}')
    else:
        return (f'{CDN_BASE}/{full_key}')
    
def put_item(table_name='PROMPT_CONFIG', data=None,host=""):
    """
    table_name: dynamo table name
    label: model label name
    sm_endpoint: model SageMaker endpoint
    """
    image_url=data["image_url"]
     
    new_url=download_and_upload(image_url)
    data["image_url"]=new_url
    #data["image_url"]=image_url
    
    prompt_sha1=hashlib.sha1((data["prompt"].strip()+","+data["image_url"].strip()).encode())
    data["prompt_sha1"]=prompt_sha1.hexdigest()
    data["PK"]="PromptConfig"
    data["publish"]="false"
    time = datetime.now()
    time_prefix=time.strftime("%Y/%m/%d")
    data["date"]=time_prefix
    ddb_resource = boto3.resource('dynamodb')
    table = ddb_resource.Table(table_name)
    resp = table.put_item(Item=data)
    return resp['ResponseMetadata']['HTTPStatusCode'] == 200

def list_prompts(table_name='PROMPT_CONFIG'):
    """
    table_name: dynamo table name
    """
    query_str = "PK = :pk "
    attributes_value={
            ":pk": {"S": "PromptConfig"},
    }
    dynamodb = boto3.client('dynamodb')
    resp = dynamodb.query(
        TableName=table_name,
        KeyConditionExpression=query_str,
        ExpressionAttributeValues=attributes_value,
        ScanIndexForward=True
    )
    items = resp.get('Items',[])
    
    time = datetime.now()
    time_prefix=time.strftime("%Y/%m/%d")
    
    prompts=[{"prompt":item["prompt"]["S"],"negative_prompt":item.get("negative_prompt",{}).get("S"),"seed":int(item["seed"].get("N",-1)),"width":int(item["width"].get("N",512),),"height":int(item["height"].get("N",512)),"model":item["model"]["S"],"image_url":item.get("image_url",{}).get("S",""),"date":item.get("date",{}).get("S",time_prefix),"prompt_sha1":item.get("prompt_sha1",{}).get("S","")} for item in items]
    return prompts

def delete_item(table_name='PROMPT_CONFIG', pk='PromptConfig', prompt_sha1=None):
    """
    table_name: dynamo table name
    prompt_sha1: prompt sha1 
    """
    if prompt_sha1 is None:
        return False
    ddb_resource = boto3.resource('dynamodb')
    table = ddb_resource.Table(table_name)
    resp = table.delete_item(
        Key={
            'PK': pk,
            'prompt_sha1': prompt_sha1
        }
    )
    return resp['ResponseMetadata']['HTTPStatusCode'] == 200
    
def search_item(table_name, pk, prefix):
    #if env local_mock is true return local config
    dynamodb = boto3.client('dynamodb')
   
    if prefix == "":
        query_str = "PK = :pk "
        attributes_value={
        ":pk": {"S": pk},
        }
    else:
       query_str = "PK = :pk and begins_with(SM_ENDPOINT, :sk) "
       attributes_value[":sk"]={"S": prefix}
    
    resp = dynamodb.query(
        TableName=table_name,
        KeyConditionExpression=query_str,
        ExpressionAttributeValues=attributes_value,
        ScanIndexForward=True
    )
    items = resp.get('Items',[])
    return items

def async_inference(input_location,sm_endpoint=None):
    """"
    :param input_location: input_location used by sagemaker endpoint async
    :param sm_endpoint: stable diffusion model's sagemaker endpoint name
    """
    if sm_endpoint is None :
        raise Exception("Not found SageMaker")
    response = sagemaker_runtime.invoke_endpoint_async(
            EndpointName=sm_endpoint,
            InputLocation=input_location)
    return response["ResponseMetadata"]["HTTPStatusCode"], response.get("OutputLocation",'')


def get_async_inference_out_file(output_location):
    """
    :param output_locaiton: async inference s3 output location
    """
    s3_resource = boto3.resource('s3')
    output_url = urllib.parse.urlparse(output_location)
    bucket = output_url.netloc
    key = output_url.path[1:]
    try:
        obj_bytes = s3_resource.Object(bucket, key)
        value = obj_bytes.get()['Body'].read()
        data = json.loads(value)
        images=data['result']
        if CDN_BASE!="":
            images=[x.replace(f"s3://{S3_BUCKET}",f"{CDN_BASE}") for x in images]
        return {"status":"completed", "images":images}
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "NoSuchKey":
            return {"status":"Pending"}
        else:
            return {"status":"Failed", "msg":"have other issue, please contact site admini"}


def result_json(status_code,body,cls=None):
    """
    :param status_code: return http status code
    :param body: return body  
    """
    if cls != None:
        body=json.dumps(body,cls=cls)
    else:
        body = json.dumps(body)
    return {
        'statusCode': status_code,
        'isBase64Encoded': False,
        'headers': {
            'Content-Type': 'application/json',
            'access-control-allow-origin': '*',
            'access-control-allow-methods': '*',
            'access-control-allow-headers': '*'
            
        },
        'body': body
    }

def get_s3_uri(bucket, prefix):
    """
    s3 url helper function
    """
    if prefix.startswith("/"):
        prefix=prefix.replace("/","",1)
    return f"s3://{bucket}/{prefix}"

def lambda_handler(event, context):
    """
    lambda main function
    """
    print(f"=========event========\n{event}\n")
    try:
        http_method=event.get("httpMethod","GET")
        request_path=event.get("path","")
        if http_method=="OPTIONS":
             return result_json(200,[])
             
        if http_method=="POST" and request_path=="/upload_handler":
            body =event.get("body","")
            if body is None:
                body=""
            else:
                body = json.loads(body)
            if "imageName" in body and "imageData" in body:
                file_content = base64.b64decode(body["imageData"])
                if "jpg" in body["imageName"] or "jpeg" in body["imageName"] :
                    file_name=f"stablediffusion/upload/{str(uuid.uuid4())}.jpg"
                    conttent_type="image/jpeg"
                else:
                    file_name=f"stablediffusion/upload/{str(uuid.uuid4())}.png"
                    conttent_type="image/png"
                # 保存文件到S3存储桶
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=file_name,
                    Body=file_content,
                    ContentType=conttent_type
                )
                return result_json(200,{"upload_file":file_name})
            else:
                return result_json(200,{'msg':'Hello from Lambda!'}) 
        if http_method=="POST" and request_path=="/async_hander":
            #check request body
            body=event.get("body","")
            if body=="":
                return result_json(400,{"msg":"need prompt"})
            sm_endpoint=event["headers"].get("x-sm-endpoint",None)
            #check sm_endpoint , if request have not check it from dynamodb
            if sm_endpoint is None:
                items=search_item(DDB_TABLE, "APIConfig", "")
                configs=[APIconfig(item) for item in items]
                if len(configs)>0:
                    sm_endpoint=configs[0].sm_endpoint
                else:
                     return result_json(400,{"msg":"not found SageMaker Endpoint"})
                
            input_file=str(uuid.uuid4())+".json"
            s3_resource = boto3.resource('s3')
            s3_object = s3_resource.Object(S3_BUCKET, f'{S3_PREFIX}/input/{input_file}')
            s3_object.put(
                Body=(bytes(body.encode('UTF-8')))
            )
            print(f'input_location: s3://{S3_BUCKET}/{S3_PREFIX}/input/{input_file}')
            status_code, output_location=async_inference(f's3://{S3_BUCKET}/{S3_PREFIX}/input/{input_file}',sm_endpoint)
            status_code=200 if status_code==202 else 403
            return result_json(status_code,{"task_id":os.path.basename(output_location).split('.')[0]})
        elif http_method=="GET" and request_path=="/config":
            print(f'HTTP/{http_method},')
            items=search_item(DDB_TABLE, "APIConfig", "")
            configs=[APIconfig(item) for item in items]
            configs=[item for item in configs if item.publish is True]
            return result_json(200,configs,cls=APIConfigEncoder)
        elif http_method=="GET" and request_path=="/prompts":
            prompts=list_prompts()
            return result_json(200,prompts)
        elif http_method=="DELETE" and request_path=="/prompts":
            body=event.get("body","")
            if body=="":
                return result_json(400,{"msg":"need prompt sha1"})
            prompts=list_prompts()
            body = json.loads(body)
            if "prompt_sha1" in body and "gallery_admin_token" in body:
                print(f'gallery_admin_token: {body["gallery_admin_token"]}, GALLERY_ADMIN_TOKEN: {GALLERY_ADMIN_TOKEN}')
                if body["gallery_admin_token"]!=GALLERY_ADMIN_TOKEN:
                    return result_json(401,{"msg":"you have no permission"})
                delete_item(prompt_sha1=body['prompt_sha1'])
                return result_json(200,{'msg':f"{body['prompt_sha1']}, remove"})
            return result_json(400,{"msg":"need prompt sha1"})
        
        elif http_method=="POST" and request_path=="/prompts":
            body=event.get("body","")
            if body=="":
                return result_json(400,{"msg":"need prompt"})
            body = json.loads(body)
            if "image_url" in body:
                key=put_item(data=body,host=event["headers"].get("origin",""))
                return result_json(200,{'msg':key})
            
            return result_json(400,{"msg":"need prompt"})
            
        elif http_method=="POST" and request_path=="/auth":
            body=event.get("body","")
            if body=="":
                return result_json(400,{"msg":"need token"})
            body = json.loads(body)
            if "token" in body:
               check=(body["token"]==GALLERY_ADMIN_TOKEN)
               if check:
                   return result_json(200,{'msg':"ok"})
               else:
                   return result_json(401,{'msg':"auth failed"})
            
            return result_json(400,{"msg":"need prompt"})
                
                
            
        elif http_method=="GET" and "/task/" in request_path:
            task_id=os.path.basename(request_path)
            if task_id!="":
                result=get_async_inference_out_file(f"s3://{S3_BUCKET}/{S3_PREFIX}/out/{task_id}.out")
                status_code=200 if result.get("status")=="completed" else 204
                return result_json(status_code,result)
            else:
                return result_json(400,{"msg":"Task id not exists"})

        return {
                    'statusCode': 200,
                    'headers':{
                     'Content-Type': 'text/html',
                    },
                    'body': 'AIGC workshop !'
                    }

    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        return result_json(502, {'msg':'Opps , something is wrong!'})
