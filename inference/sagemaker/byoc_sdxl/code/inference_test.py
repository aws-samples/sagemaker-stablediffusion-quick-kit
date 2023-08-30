import os

from pydantic import parse_obj_as


from inference import predict_fn, InferenceOpt

s3_bucket=os.environ.get('s3_bucket','')
if s3_bucket=="":
    raise Exception("need setup s3_bucket")


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
                    "control_net_enable":"enable",
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
                },
    "SDXL":  {
                "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
                "steps":20,
                "sampler":"euler_a",
                "count":1,
                "control_net_enable":"enable",
                "sdxl_refiner":"enable",
                "control_net_model":"depth",
                "input_image":"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            },
    "SDXL_CN":  {
                "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
                "steps":20,
                "sampler":"euler_a",
                "count":1,
                "control_net_enable":"enable",
                "sdxl_refiner":"enable",
                "control_net_model":"depth",
                "input_image":"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            }
}           





def test_sdxl_predict():
    assert inputs.get("SDXL",None) is not None
    data=parse_obj_as(InferenceOpt,inputs.get("SDXL",None))
    predict_fn(data)
    



test_sdxl_predict()
test_sdxl_predict()
