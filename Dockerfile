FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

# Generate dataset and train model on build
RUN python dataset_generator.py && python framing_bias_detector.py

EXPOSE 5000

# Use gunicorn for production serving, binding to the PORT env variable (default 5000)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 app:app"]
