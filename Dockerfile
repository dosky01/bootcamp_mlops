FROM python:3.10-slim
COPY requirements.txt /tmp
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN groupadd -g 1000 mlgroup && useradd -u 1000 -g mlgroup mluser
RUN mkdir -p /usr/src/app && chown mluser:mlgroup /usr/src/app
USER mluser:mlgroup

WORKDIR /usr/src/app
COPY --chown=mluser:mlgroup ./churner ./churner
COPY --chown=mluser:mlgroup ./saved ./saved
COPY --chown=mluser:mlgroup ./config.yaml ./config.yaml
COPY --chown=mluser:mlgroup ./Makefile ./Makefile

EXPOSE 5555

#RUN make train

#ENTRYPOINT ["python"]
#CMD ["gunicorn", "--bind", ":5555", "./churner/app/api:app"]
#ENTRYPOINT ["python"]
CMD ["python", "churner/app/api.py"]