from fastapi import FastAPI
from time import sleep
import uvicorn
from datetime import datetime
from fastapi import Request


from inference import model_fn,predict_fn,prepare_opt


app = FastAPI()


@app.get('/ping')
async def ping():
    return {"message": "ok"}


@app.post('/invocations')
async def invocations(request: Request):
    body=await request.json()
    result=inference_fn(body)
    return result


def inference_fn(data):
    data=prepare_opt(data)
    model=None
    return {'result':predict_fn(data,model)}


