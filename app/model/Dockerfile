FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install -y procps && pip install -U pip && rm /etc/localtime && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

RUN pip install -r ./requirements.txt

COPY ./main.py /code/

EXPOSE 8000

CMD ["uvicorn","main:app", "--host", "0.0.0.0", "--port", "8000"]