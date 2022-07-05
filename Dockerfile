FROM python:3.9.1
FROM condaforge/miniforge3

WORKDIR /sequence-mnist

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt

COPY sequence_mnist sequence_mnist
COPY tests tests