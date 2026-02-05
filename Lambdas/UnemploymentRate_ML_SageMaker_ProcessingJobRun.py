#Automated triggered Lambda function (7:40 on 21th of each month) to start a SageMaker processing job for SARIMA model training and forecasting of unemployment rate.

import boto3
import time
import os

# Create SageMaker client in your region
sagemaker = boto3.client("sagemaker", region_name="eu-central-1")

# Read environment variables
ARN_ROLE = os.environ.get("ARN_ROLE")
IMAGE_URI = os.environ.get("IMAGE_URI")

def lambda_handler(event, context):

    # Unique job name (required by SageMaker)
    timestamp = int(time.time())

    job_name = f"UnemploymentRate-ML-SageMaker-SARIMA-{timestamp}"

    response = sagemaker.create_processing_job(

        ProcessingJobName=job_name,

        # Your SageMaker execution role
        RoleArn=ARN_ROLE,

        AppSpecification={
            "ImageUri": IMAGE_URI
        },

        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.t3.large",
                "VolumeSizeInGB": 20
            }
        },

        # Max runtime: 2 hours
        StoppingCondition={
            "MaxRuntimeInSeconds": 7200
        }
    )

    return {
        "statusCode": 200,
        "job_name": job_name,
        "message": "Monthly SARIMA processing job started successfully"
    }
