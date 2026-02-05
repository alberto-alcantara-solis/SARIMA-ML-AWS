#Dockerfile for creating a custom image to run the SARIMA forecasting job in SageMaker processing job. It installs necessary Python libraries and copies the forecasting script into the image.

FROM python:3.11-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    boto3 \
    statsmodels \
    scikit-learn

# Copy script
COPY sarima_forecast_job.py /opt/program/sarima_forecast_job.py

ENTRYPOINT ["python", "/opt/program/sarima_forecast_job.py"]