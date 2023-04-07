FROM tensorflow/tensorflow:latest-gpu

# Create the environment:
RUN apt-get update
RUN apt-get update --fix-missing
RUN apt-get install -y iputils-ping iproute2 git tmux virtualenv wget vim curl

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install wandb
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install gdown tensorboard
RUN pip install yuvio
RUN pip install scikit-learn
RUN pip install seaborn
RUN pip install matplotlib

RUN pip install Pillow
RUN pip install pandas
RUN pip install opencv-python
RUN pip install lxml
RUN pip install imgaug
RUN pip install beautifulsoup4

# Add the eidoslab group to the image 
# not sure it is really needed but ok
RUN addgroup --gid 1337 eidoslab

# This is wandb stuff
RUN mkdir /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

# For pytorch checkpoints
RUN mkdir /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src


WORKDIR /src

ENTRYPOINT ["python"]