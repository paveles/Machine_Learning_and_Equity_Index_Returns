FROM python:3
SHELL ["/bin/bash", "-c"]
ADD . /code/
WORKDIR /code/
RUN apt-get update && apt-get install make
RUN make requirements
CMD [ "python","src/main.py"]
