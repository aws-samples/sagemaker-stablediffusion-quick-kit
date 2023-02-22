#命令行获取vpcid
vpcid=$(aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text)

#获取subnetid,选取2个即可
subnetids=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${vpcid}" --query 'Subnets[0:2].SubnetId' --output text)
subnetids=$(echo ${subnetids} | tr " " ",")

endpoints=$(aws sagemaker list-endpoints --query 'Endpoints[*].EndpointName' --output text)


# Color variables
red='\033[0;31m'
green='\033[0;32m'
yellow='\033[0;33m'
blue='\033[0;34m'
magenta='\033[0;35m'
cyan='\033[0;36m'
# Clear the color after that
clear='\033[0m'


echo -e "\n${clear}=======sam deploy parameter=======\n"
echo -e "VpcId:[${green}" ${vpcid} "${clear}]\n"
echo -e "Subnets:[${green}"  ${subnetids} "${clear}]\n"
echo -e "SageMakerEndpoint:[${green}"  ${endpoints} "${clear}]\n"
echo -e "${clear}\n==================================\n"






