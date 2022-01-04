FROM python:3
COPY . /app
WORKDIR /app
EXPOSE 5000
RUN pip install -r requirements.txt

CMD ["python", "python3 web.py"]
