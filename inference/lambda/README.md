## Deploy Cloudfront , Application Load Balancer , Amazon Lambda

### 1. Install sam 

 ```bash
 curl -OL https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip
 unzip aws-sam-cli-linux-x86_64.zip -d sam-installation
 sudo ./sam-installation/install
 
 #verify sam version
 sam --version
 ```



### 2. Deploy resource with SAM  


```bash
sam build
sam deploy --guided # input your configuration
```



### 3. DynamoDB config tools

ddb_util.py is config tools with dynamodb,  need boto3 library.

```bash
#list all config 
python ddb_util.py --action list

#add sagemaker endpoint to DDB
python ddb_util.py --action add --label mymodel --sm_endpoint aigc-test

#remove sagemaker endpoint from DDB
python ddb_util.py --action remove --sm_endpoint aigc-test
```

