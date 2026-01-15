ARG PYTHON_VERSION=3.11
FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.9.11 as watchdog
FROM --platform=${TARGETPLATFORM:-linux/amd64} python:${PYTHON_VERSION}
#FROM --platform=${TARGETPLATFORM:-linux/amd64} stablebaselines/stable-baselines3-cpu

ARG TARGETPLATFORM
ARG BUILDPLATFORM

ENV PIP_NO_COMPILE=1
ENV PIP_NO_BUILD_ISOLATION=1


# Allows you to add additional packages via build-arg
ARG ADDITIONAL_PACKAGE

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog
RUN apt-get update \
    && apt-get install -y ca-certificates ${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/

# Add non root user
RUN groupadd app && useradd -r -g app app

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .

RUN chown -R app /home/app && \
    mkdir -p /home/app/python && chown -R app /home/app
USER app
ENV PATH=$PATH:/home/app/.local/bin:/home/app/python/bin/
ENV PYTHONPATH=$PYTHONPATH:/home/app/python

RUN pip install -r requirements.txt --target=/home/app/python

RUN mkdir -p function
RUN touch ./function/__init__.py

WORKDIR /home/app/function/
COPY function/requirements_other.txt .
COPY function/requirements.txt .

#RUN pip install -r requirements_other.txt --target=/home/app/python
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu --target=/home/app/python
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 --no-build-isolation
RUN pip install --index-url https://pypi.org/simple -r requirements.txt --no-build-isolation
#RUN pip uninstall -y nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12
#nvidia-cublas-cu12 nvidia-cudnn-cu12

RUN pip freeze
RUN pip cache purge

WORKDIR /home/app/

USER root

COPY function           function

# Allow any user-id for OpenShift users.
RUN chown -R app:app ./ && \
    chmod -R 777 /home/app/python

USER app

ENV fprocess="python3 index.py"
EXPOSE 8080

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
