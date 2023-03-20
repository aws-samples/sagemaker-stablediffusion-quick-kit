algorithm_name=sd-inference-v2

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

#load public ECR image
#aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker pull public.ecr.aws/o7x6j3x6/sd-dreambooth-finetuning-v2

# Log into Docker
pwd=$(aws ecr get-login-password --region ${region})
docker login --username AWS -p ${pwd} ${account}.dkr.ecr.${region}.amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
mkdir -p ./sd_code/extensions
cd ./sd_code/extensions/ && git clone https://github.com/qingyuan18/sd_dreambooth_extension.git
cd ../../
docker build -t ${algorithm_name}  ./ -f ./Dockerfile.public-ecr
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
rm -rf ./sd_code
