FROM tensorflow/tensorflow
ADD ./src /app/src
ADD requirements.txt /app/src
ADD ./doc /app/doc
VOLUME /app/doc
WORKDIR /app/src
RUN pip install -r requirements.txt
