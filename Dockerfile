FROM python:3.10.11

EXPOSE 3000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .


CMD ["python", "app.py"]