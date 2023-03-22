algorithm_name=lora-finetuning-v2

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

# Log into Docker
pwd=$(aws ecr get-login-password --region ${region})
docker login --username AWS -p ${pwd} ${account}.dkr.ecr.${region}.amazonaws.com

mkdir -p ./sd_code
cp ./training/requirements_v2.txt ./sd_code/
cd ./sd_code/ && git clone https://github.com/qingyuan18/sd-scripts.git
cd ../

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build -t ${algorithm_name}  ./ -f ./dockerfile_lora_v2 > ./docker_build.log
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
rm -rf ./sd_code