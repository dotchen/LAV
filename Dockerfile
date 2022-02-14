FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy

RUN apt-get update && apt-get install --reinstall -y  locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
	     libpng16-16 \
	     libtiff5 \
         libpng-dev \
         python-dev \
         python3.5 \
         python3.5-dev \
         python-networkx \
         python-setuptools \
         python3-setuptools && \ 
         rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o- https://bootstrap.pypa.io/pip/3.5/get-pip.py | python3.5

# installing conda
RUN curl -o ~/miniconda.sh -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/bin/conda create -n python37 python=3.7 numpy networkx scipy six requests

RUN packages='py_trees==0.8.3 shapely six dictor requests ephem tabulate' \
	&& pip3 install ${packages}

WORKDIR /workspace
COPY .tmp/PythonAPI /workspace/CARLA/PythonAPI
ENV CARLA_ROOT /workspace/CARLA

ENV PATH "/workspace/CARLA/PythonAPI/carla/dist/carla-leaderboard-py3x.egg":/opt/conda/envs/python37/bin:/opt/conda/envs/bin:$PATH

# adding CARLA egg to default python environment
RUN pip install --user setuptools py_trees==0.8.3 psutil shapely six dictor requests ephem tabulate

ENV SCENARIO_RUNNER_ROOT "/workspace/scenario_runner"
ENV LEADERBOARD_ROOT "/workspace/leaderboard"
ENV TEAM_CODE_ROOT "/workspace/team_code"
ENV PYTHONPATH "/workspace/CARLA/PythonAPI/carla/dist/carla-leaderboard-py3x.egg":"${SCENARIO_RUNNER_ROOT}":"${CARLA_ROOT}/PythonAPI/carla":"${LEADERBOARD_ROOT}":${PYTHONPATH}

COPY .tmp/scenario_runner ${SCENARIO_RUNNER_ROOT}
COPY .tmp/leaderboard ${LEADERBOARD_ROOT}
COPY .tmp/team_code ${TEAM_CODE_ROOT}

RUN mkdir -p /workspace/results
RUN chmod +x /workspace/leaderboard/scripts/run_evaluation.sh


########################################################################################################################
########################################################################################################################
############                                BEGINNING OF USER COMMANDS                                      ############
########################################################################################################################
########################################################################################################################

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python pyyaml
RUN pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.1+cu101.html
RUN pip install einops==0.3.2
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx

ENV TEAM_AGENT ${TEAM_CODE_ROOT}/lav_agent.py
ENV TEAM_CONFIG ${TEAM_CODE_ROOT}/config.yaml
ENV CHALLENGE_TRACK_CODENAME SENSORS

########################################################################################################################
########################################################################################################################
############                                   END OF USER COMMANDS                                         ############
########################################################################################################################
########################################################################################################################

ENV SCENARIOS ${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
ENV ROUTES ${LEADERBOARD_ROOT}/data/routes_training.xml
ENV REPETITIONS 1
ENV CHECKPOINT_ENDPOINT /workspace/results/results.json
ENV DEBUG_CHALLENGE 0

ENV HTTP_PROXY ""
ENV HTTPS_PROXY ""
ENV http_proxy ""
ENV https_proxy ""


CMD ["/bin/bash"]
