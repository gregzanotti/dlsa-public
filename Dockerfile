FROM ubuntu:22.04


# LABEL about the custom image
LABEL maintainer="redpoint13@gmail.com"
LABEL version="0.1"
LABEL description="Deep Learning methodlogy for statistical arbitrage of liquid securities"

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

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

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
COPY start.sh /start.sh
ENTRYPOINT ["./start.sh"]

# CMD [ "/bin/bash" ]