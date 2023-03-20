import os
import diffusers

from inference import model_fn,predict_fn,prepare_opt

os.environ['s3_bucket']='sagemaker-us-east-1-596030579944'


inputs={
   "canny": {
                "prompt": "taylor swift, best quality, extremely detailed",
                "negative_prompt":"monochrome, lowres, bad anatomy, worst quality, low quality",
                "steps":20,
                "sampler":"euler_a",
                "seed":43768,
                "height": 512, 
                "width": 512,
                "count":2,
                "control_net_model":"canny",
                "input_image":"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png",
                },
    "openpose":    {
                    "prompt": "super-hero character, best quality, extremely detailed",
                    "negative_prompt":"monochrome, lowres, bad anatomy, worst quality, low quality",
                    "steps":20,
                    "sampler":"euler_a",
                    "seed":43768,
                    "height": 512,
                    "width": 512,
                    "count":2,
                    "control_net_model":"openpose",
                    "input_image":"https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/yoga1.jpeg"
                },
     "mlsd":    {
                    "prompt": "room",
                    "negative_prompt":"monochrome, lowres, bad anatomy, worst quality, low quality",
                    "steps":20,
                    "sampler":"euler_a",
                    "seed":43768,
                    "height": 512,
                    "width": 512,
                    "count":2,
                    "control_net_model":"mlsd",
                    "input_image":"https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png"
                },
    "depth":            {
                    "prompt": "Stormtrooper's lecture",
                    "negative_prompt":"monochrome, lowres, bad anatomy, worst quality, low quality",
                    "steps":20,
                    "sampler":"euler_a",
                    "seed":43768,
                    "height": 512,
                    "width": 512,
                    "count":2,
                    "control_net_model":"depth",
                    "input_image":"https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
                },
    
   "hed":             {
                    "prompt": "oil painting of handsome old man, masterpiece",
                    "negative_prompt":"monochrome, lowres, bad anatomy, worst quality, low quality",
                    "steps":20,
                    "sampler":"euler_a",
                    "seed":43768,
                    "height": 512,
                    "width": 512,
                    "count":2,
                    "control_net_model":"hed",
                    "input_image":"https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/images/man.png"
                },
    "scribble":  {
                    "prompt": "bag",
                    "negative_prompt":"monochrome, lowres, bad anatomy, worst quality, low quality",
                    "steps":20,
                    "sampler":"euler_a",
                    "seed":43768,
                    "height": 512,
                    "width": 512,
                    "count":2,
                    "control_net_model":"scribble",
                    "input_image":"https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png"
                }
}           

model=model_fn(".")

def test_model_fn():    
    assert isinstance(model,diffusers.DiffusionPipeline)
    
    
# def test_canny_predict():
#     assert inputs.get("canny",None) is not None
#     data=prepare_opt(inputs.get("canny",None))
#     predict_fn(data,model)
    
    
def test_openpose_predict():
    assert inputs.get("openpose",None) is not None
    data=prepare_opt(inputs.get("openpose",None))
    predict_fn(data,model)
    
# def test_mlsd_predict():
#     assert inputs.get("mlsd",None) is not None
#     data=prepare_opt(inputs.get("mlsd",None))
#     predict_fn(data,model)
    
    
# def test_depth_predict():
#     assert inputs.get("depth",None) is not None
#     data=prepare_opt(inputs.get("depth",None))
#     predict_fn(data,model)
    
# def test_hed_predict():
#     assert inputs.get("hed",None) is not None
#     data=prepare_opt(inputs.get("hed",None))
#     predict_fn(data,model)
    
   
# def test_scribble_predict():
#     assert inputs.get("scribble",None) is not None
#     data=prepare_opt(inputs.get("scribble",None))
#     predict_fn(data,model)
    
def test_openpose_02_predict():
    assert inputs.get("openpose",None) is not None
    data=prepare_opt(inputs.get("openpose",None))
    predict_fn(data,model)

test_openpose_02_predict()
test_openpose_02_predict()
