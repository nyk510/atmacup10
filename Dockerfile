FROM registry.gitlab.com/nyker510/analysis-template/cpu:1.0.4

RUN pip install -U pip && \
  pip install \
    python-vivid==0.3.3.4 \
    shortuuid \
    interpret \
    pygam \
    dataclasses_json \
    texthero \
    pip install git+https://gitlab+deploy-token-373496:JZxuUxVmg682HGji1Zfs@gitlab.com/atma_inc/anemone.git \
    pandas==1.2.2

WORKDIR /home/penguin
RUN git clone https://github.com/facebookresearch/fastText.git && \
  cd fastText && \
  pip install . && \
  rm -rf /home/penguin/fastText
