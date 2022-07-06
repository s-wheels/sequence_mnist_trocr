FROM python:3.9.1

WORKDIR /sequence-mnist

# Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml -n seqmn_huggingface

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY sequence_mnist sequence_mnist
COPY tests tests