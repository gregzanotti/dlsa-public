# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:1.12.3-gpu-py3

# LABEL about the custom image
LABEL maintainer="redpoint13@gmail.com"
LABEL version="0.1"
LABEL description="Deep Learning methodlogy for statistical arbitrage of liquid securities"

SHELL [ "/bin/bash", "-c" ]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
# ARG PYTHON_VERSION_TAG=3.9
# ARG LINK_PYTHON_TO_PYTHON3=1
# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

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
        tree \
        # python3 \
        # python3-pip \
        # python3-venv \
        software-properties-common && \
    mv /usr/bin/lsb_release /usr/bin/lsb_release.bak && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# ENV MPLCONFIGDIR /.config/matplotlib
# ENV VIRTUAL_ENV=/opt/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN python3 -m venv $VIRTUAL_ENV

# Enable tab completion by uncommenting it from /etc/bash.bashrc
# The relevant lines are those below the phrase "enable bash completion in interactive shells"
RUN export SED_RANGE="$(($(sed -n '\|enable bash completion in interactive shells|=' /etc/bash.bashrc)+1)),$(($(sed -n '\|enable bash completion in interactive shells|=' /etc/bash.bashrc)+7))" && \
    sed -i -e "${SED_RANGE}"' s/^#//' /etc/bash.bashrc && \
    unset SED_RANGE

# COPY requirements.txt requirements.txt
# RUN pip install --upgrade pip venv
# RUN pip install -r requirements.txt

# EXPOSE 3000
COPY . .
# CMD ["flask", "run"]
# Copy start.sh script and define default command for the container
# COPY start.sh /start.sh
ENTRYPOINT ["./start.sh"]
# ENTRYPOINT [ "/bin/bash" ]

