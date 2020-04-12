FROM python:3.6.10-buster

# pip install first to improve caching in Docker image.
RUN  pip install packaging && \
     pip install git+git://github.com/rbracco/fastai2_audio.git

# Install Dependencies.
RUN apt update && apt install -y wget unzip libsndfile1-dev sox
# Add the source to image.
ADD . /app

# Switch to source directory.
WORKDIR /app

# Install Python requirements and dependencies.
RUN pip install -r requirements.txt

# Install the assignment.
RUN python setup.py install

# Print help and enter bash.
CMD inference --help && bash

# To build the container:
# $ docker build -t audio_deep_learning:latest .

# To run the container:
# $ docker run -v path/to/host/folder:/app/data --rm -it audio_deep_learning
