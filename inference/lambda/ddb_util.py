#!/usr/bin/env python
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

import sys 
import argparse
import traceback



import boto3
from botocore.exceptions import ClientError


ddb_client = boto3.client('dynamodb')
ddb_resource = boto3.resource('dynamodb')

sm_client = boto3.client('sagemaker')

def list_item(table_name='AIGC_CONFIG'):
    """
    table_name: dynamo table name
    """
    query_str = "PK = :pk "
    attributes_value={
            ":pk": {"S": "APIConfig"},
    }
    resp = ddb_client.query(
        TableName=table_name,
        KeyConditionExpression=query_str,
        ExpressionAttributeValues=attributes_value,
        ScanIndexForward=True
    )
    items = resp.get('Items',[])
    
    configs=[{"label":item["LABEL"]["S"],"sm_endpoint":item["SM_ENDPOINT"]["S"],"hit":item.get("HIT",{}).get("S","")} for item in items]
    return configs

def put_item(table_name='AIGC_CONFIG', label=None,sm_endpoint=None,hit=''):
    """
    table_name: dynamo table name
    label: model label name
    sm_endpoint: model SageMaker endpoint
    """
    item = {"LABEL":label, "SM_ENDPOINT": sm_endpoint,"HIT":hit, "PK": "APIConfig"}
    table = ddb_resource.Table(table_name)
    resp = table.put_item(Item=item)
    return resp['ResponseMetadata']['HTTPStatusCode'] == 200

def delete_item(table_name='AIGC_CONFIG', pk='APIConfig', sm_endpoint=None):
    """
    table_name: dynamo table name
    sm_endpoint: model SageMaker endpoint
    """
    if sm_endpoint is None:
        return False
    table = ddb_resource.Table(table_name)
    resp = table.delete_item(
        Key={
            'PK': pk,
            'SM_ENDPOINT': sm_endpoint
        }
    )
    return resp['ResponseMetadata']['HTTPStatusCode'] == 200

def check_dynamodb_table(table_name):
    """
    table_name: dynamo table name
    """
    table = ddb_resource.Table(table_name)
    try:
        print(table.table_status)
        return True
    except ClientError:
        return False
    
def check_sm_endpoint(sm_endpoint=None):
    """
    sm_endpoint: model SageMaker endpoint
    """
    try:
        response = sm_client.describe_endpoint(
        EndpointName=sm_endpoint
        )
        print(response)
        return True
    except ClientError:
        return False
    
def create_dynamodb_table(table_name):
    """
    table_name: dynamo table name
    """
    if check_dynamodb_table(table_name):
        print(f"Table [{table_name}] already exists")
        return False
    table = ddb_resource.create_table(
        TableName=table_name,
        AttributeDefinitions=[
            {
                "AttributeName": "PK",
                "AttributeType": "S"
            },
            {
                "AttributeName": "SM_ENDPOINT",
                "AttributeType": "S"
            }
        ],
        KeySchema=[
            {
                "AttributeName": "PK",
                "KeyType": "HASH"
            },
            {
                "AttributeName": "SM_ENDPOINT",
                "KeyType": "RANGE"
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 2,
            'WriteCapacityUnits': 2
        }
    )
    print(f"Create table [{table_name}] completed ")
    return True


def main():
    """
    main function 
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--action', dest='action',
                        type=str,
                        required=True,
                        default='list',
                        help='list|add|remove|create_table')
    parser.add_argument('--sm_endpoint',
                        type=str,
                        default=None,
                        help='sagemaker endpoint')
    parser.add_argument('--label',
                        type=str,
                        default=None,
                        help='label')
    parser.add_argument('--hit',
                        type=str,
                        default='',
                        help='style hit')
    parser.add_argument('--ddb_table',
                        type=str,
                        default='AIGC_CONFIG',
                        help='dynamo table name,default AIGC_CONFIG')
    
    args = parser.parse_args()

    action = args.action.lower()
    
    if action=='list':
        configs=list_item(args.ddb_table)
        for conf in configs:
            print(f'label: {conf["label"]}, sm_endpoint: {conf["sm_endpoint"]} , hit: {conf["hit"]}')
    elif action == 'add':
        if args.label is None or args.sm_endpoint is None:
            print('You must input label and sagemaker endpoint')
            return
        endpoint_exists=check_sm_endpoint(args.sm_endpoint)
        if endpoint_exists is False:
            print(f"Add failed , SageMaker Endpoint [{args.sm_endpoint}] not exists")
        else:
            add_action=put_item(table_name=args.ddb_table,label=args.label,sm_endpoint=args.sm_endpoint,hit=args.hit)
            print(f"Add : { add_action}")
    elif action == 'remove':
        if args.sm_endpoint is None:
            print('You must input sagemaker endpoint')
            return
        remove_action=delete_item(table_name=args.ddb_table,sm_endpoint=args.sm_endpoint)
        print(f"Remove action : { remove_action}")
    elif action == 'create_table':
        create_dynamodb_table(args.ddb_table)
    else:
        print(f'Not support action: {action}')

if __name__== "__main__":
    main()