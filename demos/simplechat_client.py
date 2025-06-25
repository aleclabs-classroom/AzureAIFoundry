from dotenv import load_dotenv

import os
from openai import AzureOpenAI

load_dotenv() # read from a .env file

endpoint = "https://eus2foundry.openai.azure.com/"

deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
api_key=os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")

subscription_key = api_key
api_version = api_version

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What are the first things to do after the tutorial in TotK?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)