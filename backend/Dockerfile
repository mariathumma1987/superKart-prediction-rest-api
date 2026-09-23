# Use an official lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency definition file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching build index
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script and serialized model into container
COPY app.py .
COPY superkart_model.joblib .

# Expose Flask API port
EXPOSE 5000

# Run Flask application using Gunicorn WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:superkart_sales_predictor_api"]
