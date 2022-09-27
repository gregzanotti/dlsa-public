ARG UBUNTU_VER=20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9

FROM ubuntu:${UBUNTU_VER}
# System packages
# LABEL about the custom image
LABEL maintainer="redpoint13@gmail.com"
LABEL version="0.1"
LABEL description="Deep Learning methodlogy for asset pricing of liquid securities"
SHELL [ "/bin/bash", "-c" ]

RUN apt-get -qq -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        gcc \
        g++ \
        wget \
        curl \
        git \
        make \
        sudo \
        bash-completion \
        build-essential \
        tree \
        software-properties-common && \
        apt-get update && \
        apt-get -y autoclean && \
        apt-get -y autoremove && \
        rm -rf /var/lib/apt/lists/*
# Use the above args 
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init

# Enable tab completion by uncommenting it from /etc/bash.bashrc
# The relevant lines are those below the phrase "enable bash completion in interactive shells"
RUN export SED_RANGE="$(($(sed -n '\|enable bash completion in interactive shells|=' /etc/bash.bashrc)+1)),$(($(sed -n '\|enable bash completion in interactive shells|=' /etc/bash.bashrc)+7))" && \
    sed -i -e "${SED_RANGE}"' s/^#//' /etc/bash.bashrc && \
    unset SED_RANGE

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install linter Black
RUN pip install black

# COPY requirements.txt requirements.txt
# RUN pip install --upgrade pip venv
# RUN pip install -r requirements.txt

# EXPOSE 3000
COPY . .
# CMD ["flask", "run"]

# Copy start.sh script and define default command for the container
# COPY start.sh /start.sh
# ENTRYPOINT ["./start.sh"]
CMD [ "/bin/bash" ]