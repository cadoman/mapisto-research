FROM jupyter/scipy-notebook
USER root
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev
USER jovyan
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt