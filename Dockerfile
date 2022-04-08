FROM python:3.9
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
WORKDIR /app
ENTRYPOINT ["streamlit","run"]
CMD SA.py
