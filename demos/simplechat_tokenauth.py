import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

load_dotenv() # read from a .env file

openai_endpoint=os.getenv("OPENAI_ENDPOINT")
api_key=os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model_deployment_name=os.getenv("MODEL_DEPLOYMENT_NAME")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
    
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint = openai_endpoint, 
    azure_ad_token_provider=token_provider
    )

deployment_name=model_deployment_name #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 

# Send a completion call to generate an answer
print('Sending a test completion job')
# start_phrase = 'Write a tagline for an ice cream shop. '
# response = client.chat.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=10)
start_phrase=""
prompts=[
            {
                "role": "system",
                "content": "You are an unhelpful assistant.",
            },
            {
                "role": "user",
                "content": "My character is level 50 in Skyrim. What should I do?",
            },
        ]
response = client.chat.completions.create(
    messages=prompts,
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment_name
)
print(start_phrase+response.choices[0].message.content)