FROM jupyter/base-notebook:python-3.8.8

USER root
ENV DEBIAN_FRONTEND=noninteractive

# load the source
COPY . /opt/debater
WORKDIR /opt/debater

# build Python
RUN apt-get -y update
RUN apt-get install --no-install-recommends -y python3.8 python3-pytest python3.8-distutils python3.8-dev python3.8-venv
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /opt/debater/venv
RUN source /opt/debater/venv/bin/activate

RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN python3.8 get-pip.py
RUN python3.8 -m pip install -U setuptools wheel

# set up the Project Debater API
RUN python3.8 -m pip install -e .

# launch JupyterLab
EXPOSE 8888/tcp
USER $NB_UID
CMD jupyter-lab
