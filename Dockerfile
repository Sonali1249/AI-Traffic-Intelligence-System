# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including .streamlit folder)
COPY . ./

# Expose Streamlit default port
EXPOSE 8501

# Set environment variable for Streamlit to run in production mode
ENV STREAMLIT_SERVER_HEADLESS=true

# Entry point – run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
