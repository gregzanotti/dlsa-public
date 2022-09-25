# FROM ubuntu:22.04
FROM rapidsai/rapidsai

# SHELL [ "/bin/bash", "-c" ]

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
        software-properties-common && \
        apt-get update && \
        apt-get -y autoclean && \
        apt-get -y autoremove && \
        rm -rf /var/lib/apt/lists/*

# Install Minicond
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH

# Enable tab completion by uncommenting it from /etc/bash.bashrc
# The relevant lines are those below the phrase "enable bash completion in interactive shells"
RUN export SED_RANGE="$(($(sed -n '\|enable bash completion in interactive shells|=' /etc/bash.bashrc)+1)),$(($(sed -n '\|enable bash completion in interactive shells|=' /etc/bash.bashrc)+7))" && \
    sed -i -e "${SED_RANGE}"' s/^#//' /etc/bash.bashrc && \
    unset SED_RANGE

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD [ "/bin/bash" ]
