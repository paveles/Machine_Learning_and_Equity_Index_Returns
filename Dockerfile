FROM python:3
SHELL ["/bin/bash", "-c"]
ADD . /code/
WORKDIR /code/
RUN pip install -r requirements.txt 
RUN python src/data.py
RUN python src/train.py
RUN python src/visualize.py
