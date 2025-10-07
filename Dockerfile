FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

#To temporarily install other libraries inside the container
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /.local && chmod 777 /.local
RUN mkdir -p /.config && chmod 777 /.config

#Can change to python main.py
CMD ["bash"]