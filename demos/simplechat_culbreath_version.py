# import os
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

# this section is my own
from dotenv import load_dotenv
import os

# this section is my own
load_dotenv() # read from a .env file
openai_endpoint=os.getenv("OPENAI_ENDPOINT")
api_key=os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model_deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")

# Azure OpenAI endpoint setup
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=openai_endpoint,
    api_key=api_key,
)

# Making a simple completion request
response = client.chat.completions.create(
    model=model_deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Azure AI services briefly"},
    ]
)

print(response.choices[0].message.content)