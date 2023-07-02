FROM python:latest

COPY  main.py .

WORKDIR /app


CMD ["python", "./main.py"]