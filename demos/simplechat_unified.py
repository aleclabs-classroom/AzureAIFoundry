from dotenv import load_dotenv
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

load_dotenv() # read from a .env file

project_endpoint=os.getenv("PROJECT_ENDPOINT")
api_version=os.getenv("API_VERSION")
model_deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)

print(
    "Get an authenticated Azure OpenAI client for the parent AI Services resource, and perform a chat completion operation:"
)
with project_client.inference.get_azure_openai_client(api_version=api_version) as client:

    response = client.chat.completions.create(
        model=model_deployment_name,
        messages=[
            {
                "role": "user",
                "content": "How many feet are in a mile?",
            },
        ],
    )

    print(response.choices[0].message.content)