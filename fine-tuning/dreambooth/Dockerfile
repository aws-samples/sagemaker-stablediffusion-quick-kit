
FROM ghcr.io/allenai/pytorch:1.12.1-cuda11.3-python3.9-v1.2.0
ENV PATH="/opt/ml/code:${PATH}"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install --assume-yes apt-utils -y

RUN apt update
RUN echo "Y"|apt install vim
RUN apt install wget git -y
RUN apt install libgl1-mesa-glx -y
RUN pip install opencv-python-headless

# install the SageMaker Training Toolkit 
RUN pip3 install sagemaker-training


#install xformers from sourcecode
RUN export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6" && export FORCE_CUDA="1" && pip install ninja triton==2.0.0.dev20221120 && git clone https://github.com/xieyongliang/xformers.git /opt/ml/code/repositories/xformers && cd /opt/ml/code/repositories/xformers && git submodule update --init --recursive && pip install -r requirements.txt && pip install -e . 

RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

COPY train_dreambooth.py /opt/ml/code/
COPY ./sd_code/ /opt/ml/code/

RUN pip install -r /opt/ml/code/extensions/sd_dreambooth_extension/requirements.txt

#install s5cmd
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz | tar -xz && mv s5cmd /opt/conda/bin/


# this environment variable is used by the SageMaker training toolkit to determine our program entry point
ENV SAGEMAKER_PROGRAM train_dreambooth.py

ENTRYPOINT []
