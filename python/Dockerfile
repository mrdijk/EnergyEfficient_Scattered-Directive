FROM mdijk/hfl-base:latest

ARG NAME
ARG WHEEL_NAME
ENV NAME=$NAME

WORKDIR /app
RUN mkdir -p $NAME datasets protofiles

# Install dynamos wheel FIRST
COPY $WHEEL_NAME /tmp/
RUN pip install /tmp/$WHEEL_NAME && rm /tmp/$WHEEL_NAME

# Install other requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY hfl-data/svhn_dataset.py /app/$NAME/

COPY ./*.py /app/$NAME/
COPY ./protofiles/*.py /app/protofiles/

ENV ENV="PROD"
ENV PYTHONPATH="/app/protofiles"
CMD ["sh", "-c", "python /app/${NAME}/main.py"]
