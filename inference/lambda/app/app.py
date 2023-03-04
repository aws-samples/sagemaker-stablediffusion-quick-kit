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

import boto3

from json import JSONEncoder

from botocore.exceptions import ClientError

CURRENT_REGION= boto3.session.Session().region_name




SM_REGION=os.environ.get("SM_REGION") if os.environ.get("SM_REGION")!="" else CURRENT_REGION
S3_BUCKET=os.environ.get("S3_BUCKET","")
S3_PREFIX=os.environ.get("S3_PREFIX","stablediffusion/asyncinvoke")

DDB_TABLE=os.environ.get("DDB_TABLE","") #dynamodb table name


print(f"CURRENT_REGION |{CURRENT_REGION}|")
print(f"SM_REGION |{SM_REGION}|")
print(f"S3_BUCKET |{S3_BUCKET}|")
print(f"S3_PREFIX |{S3_PREFIX}|")



sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=SM_REGION)
s3_client = boto3.client("s3")

class APIconfig:

    def __init__(self, item,include_attr=True):
        if include_attr:
            self.sm_endpoint = item.get('SM_ENDPOINT').get('S')
            self.label = item.get('LABEL').get('S'),
            self.hit = item.get('HIT',{}).get('S','')
        else:
            self.sm_endpoint = item.get('SM_ENDPOINT')
            self.label = item.get('LABEL') 
            self.hit = item.get('HIT','')


    def __repr__(self):
        return f"APIconfig<{self.label} -- {self.sm_endpoint}>"

class APIConfigEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__
            
            
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
        images=[x.replace(f"s3://{S3_BUCKET}",f"") for x in images]
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
    print(f"=========event========\n{event}")
    try:
        http_method=event.get("httpMethod","GET")
        request_path=event.get("path","")
        print("http_method:{}".format(http_method))
        print("request_path:{}".format(request_path))
        if http_method=="OPTIONS":
             return result_json(200,[])
        if http_method=="POST" and request_path=="/async_hander":
            #check request body
            body=event.get("body","")
            print("body:{}, {}".format(body, type(body)))
            encoded_body = bytes(body.encode('UTF-8'))
            print("encoded_body:{}, {}".format(encoded_body, type(encoded_body)))
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
            return result_json(200,configs,cls=APIConfigEncoder)
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
        print("exception : {}".format(ex))
        traceback.print_exc(file=sys.stdout)
        return result_json(502, {'msg':'Opps , something is wrong!'})
