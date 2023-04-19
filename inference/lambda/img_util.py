import base64
import json

import requests


image_file = "../../images/test.png"

api_url="https://4honb6i2ke.execute-api.us-east-1.amazonaws.com/default/imageupload"
base_cdn="https://dfjcgkift2mhn.cloudfront.net"
data = {
        "image": {
            "mime": "image/png"
        }
       }
with open(image_file, "rb") as f:
    image_encode = base64.b64encode(f.read())
    data["image"]["data"]=image_encode.decode('utf-8')
    response = requests.post(api_url, json=data)
    
    if response.status_code==200:
        result=json.loads(response.content)
        if "upload_file" in result:
            print(f'{base_cdn}/{result["upload_file"]}')
    else:
        print(f"error, return code {response.status_code}")

