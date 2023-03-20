from controle_net import ControlNetDectecProcessor

from PIL import Image

processor=ControlNetDectecProcessor()

def test_init():
    
    assert type(processor) is ControlNetDectecProcessor
    

def test_canny():
    image=processor.detect_process("canny","https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    assert type(image) is Image.Image
    
    
def test_mlsd():
    image=processor.detect_process("mlsd","https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    assert type(image) is Image.Image
    
    
def test_hed():
    image=processor.detect_process("hed","https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    assert type(image) is Image.Image

    
def test_openpose():
    image=processor.detect_process("openpose","https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    assert type(image) is Image.Image
    

def test_scribble():
    image=processor.detect_process("scribble","https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    assert type(image) is Image.Image