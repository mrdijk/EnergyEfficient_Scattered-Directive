FROM mdijk/hfl-base:latest
ARG NAME
ARG WHEEL_NAME
ENV NAME=$NAME
WORKDIR /app

RUN mkdir -p ${NAME} datasets protofiles hfl_data

# Copy datasets to the containers
COPY datasets/*.mat /app/datasets/

# Install dynamos wheel - it's in the current directory now
COPY ${WHEEL_NAME} /tmp/
RUN pip install /tmp/${WHEEL_NAME} && rm /tmp/${WHEEL_NAME}

# Install requirements - it's in current directory
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy hfl-data package - it's been copied to current directory
COPY hfl-data/__init__.py /app/hfl_data/
COPY hfl-data/svhn_dataset.py /app/hfl_data/

# Copy application files from current directory
COPY *.py /app/${NAME}/

# Copy protofiles
COPY protofiles/*.py /app/protofiles/

ENV ENV="PROD"
ENV PYTHONPATH="/app/protofiles:/app"

CMD ["sh", "-c", "python /app/${NAME}/main.py"]
