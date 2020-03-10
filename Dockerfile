FROM python:3.6.10-stretch
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN  python -c "import nltk; nltk.download('stopwords')"
RUN  python -c "import nltk; nltk.download('punkt')"
COPY application /app
COPY database /usr/local/lib/python3.6/site-packages/database
COPY twitter /usr/local/lib/python3.6/site-packages/twitter
COPY twitter_keys.py /usr/local/lib/python3.6/site-packages/

EXPOSE 5000
ENTRYPOINT ["python", "app.py"]