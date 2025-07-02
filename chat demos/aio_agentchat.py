"""
taken from https://github.com/Azure/azure-sdk-for-python/blob/azure-ai-projects_1.0.0b11/sdk/ai/azure-ai-agents/samples/agents_async/sample_agents_basics_async.py
"""
from dotenv import load_dotenv

import asyncio
import time

from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import MessageTextContent, ListSortOrder
from azure.identity.aio import DefaultAzureCredential

import os

load_dotenv() # read from a .env file

project_endpoint=os.environ["PROJECT_ENDPOINT"]
model_deployment_name=os.environ["MODEL_DEPLOYMENT_NAME"]

async def main() -> None:

    async with DefaultAzureCredential() as creds:
        agents_client = AgentsClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=creds,
        )

        async with agents_client:
            agent = await agents_client.create_agent(
                model=os.environ["MODEL_DEPLOYMENT_NAME"], name="my-agent", instructions="You are helpful agent"
            )
            print(f"Created agent, agent ID: {agent.id}")

            thread = await agents_client.threads.create()
            print(f"Created thread, thread ID: {thread.id}")

            message = await agents_client.messages.create(
                thread_id=thread.id, role="user", content="Hello, tell me a joke"
            )
            print(f"Created message, message ID: {message.id}")

            run = await agents_client.runs.create(thread_id=thread.id, agent_id=agent.id)

            # Poll the run as long as run status is queued or in progress
            while run.status in ["queued", "in_progress", "requires_action"]:
                # Wait for a second
                time.sleep(1)
                run = await agents_client.runs.get(thread_id=thread.id, run_id=run.id)
                print(f"Run status: {run.status}")

            if run.status == "failed":
                print(f"Run error: {run.last_error}")

            await agents_client.delete_agent(agent.id)
            print("Deleted agent")

            messages = agents_client.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.ASCENDING,
            )
            async for msg in messages:
                last_part = msg.content[-1]
                if isinstance(last_part, MessageTextContent):
                    print(f"{msg.role}: {last_part.text.value}")


if __name__ == "__main__":
    asyncio.run(main())