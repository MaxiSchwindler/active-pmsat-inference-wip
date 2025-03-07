FROM python:3.11
LABEL authors="maximilian rindler"

# set environment variables to avoid interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_FRONTEND=noninteractive

# clone active-pmsat-inference-wip
WORKDIR /git
RUN git clone --depth=1 https://github.com/MaxiSchwindler/active-pmsat-inference-wip.git

# clone pmsat-inference
RUN git clone --depth 1 --filter=blob:none --no-checkout https://gitlab.com/MaxiSchwindler/pmsat-inference.git \
    && cd pmsat-inference \
    && git sparse-checkout init --cone \
    && git sparse-checkout set pmsatlearn \
    && git checkout main

# clone AALpy
RUN git clone --depth 1 --filter=blob:none --no-checkout https://github.com/MaxiSchwindler/AALpy.git \
    && cd AALpy \
    && git sparse-checkout init --cone \
    && git sparse-checkout set aalpy \
    && git checkout master

# set PYTHONPATH
ENV PYTHONPATH="/git/active-pmsat-inference-wip:/git/pmsat-inference:/git/AALpy:${PYTHONPATH}"

# install Python dependencies
WORKDIR /git/active-pmsat-inference-wip
RUN pip install --no-cache-dir -r active_pmsatlearn/requirements.txt
RUN pip install --no-cache-dir -r evaluation/requirements.txt

# default command
CMD ["/bin/bash"]
