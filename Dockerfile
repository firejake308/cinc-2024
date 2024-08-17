FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

# Install libgl and ffmpeg
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx ffmpeg vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r ecg-image-generator/requirements.txt --no-cache-dir

CMD ["bash"]
