# Use an official lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency definition file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies without caching build index
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script into container
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
