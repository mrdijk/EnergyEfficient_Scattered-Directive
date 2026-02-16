SHELL := /bin/bash
targets := sql-query hfl-train-model hfl-train
datasets := "../configuration/datasets/"
branch_name := $(shell git rev-parse --abbrev-ref HEAD)
dockerhub_account := mdijk

.PHONY: proto hfl-base $(targets) dynamos all

proto:
	@python3 -m grpc_tools.protoc -I../proto-files --python_out=./dynamos-python-lib/protofiles --grpc_python_out=./dynamos-python-lib/protofiles ../proto-files/*.proto

# Build base image
hfl-base:
	DOCKER_BUILDKIT=1 docker build \
		--platform linux/amd64 \
		-t $(dockerhub_account)/hfl-base:$(branch_name) \
		-t $(dockerhub_account)/hfl-base:latest \
		./hfl-base/
	docker push $(dockerhub_account)/hfl-base:$(branch_name)
	docker push $(dockerhub_account)/hfl-base:latest

dynamos:
	(cd ./dynamos-python-lib && python3 setup.py bdist_wheel)

# Set PYTHONPATH to protofiles directory for includes to GRPC code to work
$(targets): proto
	cp Dockerfile ./$@

	# Build the wheel for the local package
	(cd ./dynamos-python-lib && python3 setup.py bdist_wheel)

	# Copy the wheel to the target directory
	cp ./dynamos-python-lib/dist/*.whl ./$@
	cp -r ./dynamos-python-lib/protofiles ./$@
	cp -r ./hfl-data ./$@

	# Build Docker container
	@trap 'rm -f ./$@/Dockerfile; rm -rf ./$@/*.whl; rm -rf ./$@/protofiles' EXIT;  \
	WHEEL_NAME=$$(ls ./$@/*.whl); \
	BASE=$$(basename $$WHEEL_NAME); \
	DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --build-arg NAME=$@ -t $(dockerhub_account)/$@:$(branch_name) --build-arg WHEEL_NAME=$$BASE -t $(dockerhub_account)/$@:latest ./$@/

	# Push to Docker
	docker push $(dockerhub_account)/$@:$(branch_name)
	docker push $(dockerhub_account)/$@:latest
all: all $(targets) $(proto)
