https://microsoftlearning.github.io/mslearn-ai-studio/ (tutorials)

## Steps to create a project in Azure AI Foundry and use the SDK
1. Login to portal.azure.com
    1. Create an account (with email alias for trial credits)
    2. Create a subscription
2. Deploy an AI Foundry resource and project (choose a project region that supports your model, e.g. eastus2)
    1. Name the resources (e.g., ai-rg1, eus2Foundry, eus2Project)
    2. Public network access
    3. Assign RBAC project control to user
        1. Choosing system assigned grants the AI User role (elevate to *Manager or *Owner in IAM later)
    4. Default encryption using MS managed keys
    5. No tags
    6. Create
3. Open the Foundry portal (ai.azure.com)
4. Deploy a model (e.g. pt-4o-mini in eastus2)
    * Learn what models perform which tasks (https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)
    * Tested documentation with Unified Projects (https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme?view=azure-python-preview)
    * Follow the sdk instructions on the details tab of the model
    * Click API documentation on Foundry page for instructions  (https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview?pivots=programming-language-python)
5. Open VS Code (local or https://vscode.dev)
6. Create a python env (prefer conda)
    1. Install anaconda with brew install --cask anaconda
    2. From vscode create a .py file to prompt install python and create an env
7. pip requirements
    * pip install openai azure-ai-projects azure-identity
    * pip install python-dotenv aiohttp
8. Keep secrets (e.g. API key in a .env file, add to .gitignore) (https://stackoverflow.com/questions/40216311/reading-in-environment-variables-from-an-environment-file)
9. Establish a git repo
    1. https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup
