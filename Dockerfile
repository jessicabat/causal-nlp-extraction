FROM zjunlp/oneke:v4
RUN pip install neo4j
WORKDIR /app
CMD ["/bin/bash"]
RUN apt-get update && apt-get install -y jupyter \
    && ln -s $(which jupyter-notebook) /usr/local/bin/start-notebook.sh