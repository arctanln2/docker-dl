FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN mkdir -p /opt/mnist

ENV OMP_NUM_THREADS=16

WORKDIR /opt/mnist/src
ADD main.py /opt/mnist/src/main.py

RUN  chgrp -R 0 /opt/mnist \
  && chmod -R g+rwX /opt/mnist



# ENTRYPOINT ["python3", "-m torch.distributed.run", "--nproc_per_node=2", "/opt/cifar/src/main.py"]