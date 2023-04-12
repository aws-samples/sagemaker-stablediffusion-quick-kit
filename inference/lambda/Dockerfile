FROM public.ecr.aws/lambda/python:3.10
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.6.4 /lambda-adapter /opt/extensions/lambda-adapter


ENV PORT=8080
WORKDIR /var/task
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
COPY *.py ./
ENTRYPOINT exec uvicorn --port=$PORT main:app