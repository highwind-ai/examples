FROM  --platform=linux/amd64 python:3.9-bookworm

# Add training data
RUN mkdir /data
COPY ./_local_test/data /data