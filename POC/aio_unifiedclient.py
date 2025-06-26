from dotenv import load_dotenv
import asyncio
import os

from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv() # read from a .env file

# global variables
project_endpoint=os.getenv("PROJECT_ENDPOINT")
api_version=os.getenv("API_VERSION")
model_deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)