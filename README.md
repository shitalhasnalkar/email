# AI Email Agent

An AI-powered email assistant built using Python, Streamlit, LangGraph, and Ollama.

## Features

- Email classification by intent and urgency.
- Email topic and summary generation.
- Automatic draft reply generation using an LLM.
- Human-in-the-loop review and approval.
- Edit, approve, or reject generated replies.
- Local LLM inference using Ollama and Llama 3.2.

## Technology Stack

- Python
- Streamlit
- LangGraph
- LangChain
- Ollama
- Llama 3.2

## Project Structure

```text
email-langgraph-agent/
  app.py
  email_agent.py
  requirements.txt
  README.md
```

## Installation

1. Install Python and Ollama.
2. Create and activate a virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```cmd
pip install streamlit langgraph langchain-core langchain-ollama
```

4. Download the Ollama model:

```cmd
ollama pull llama3.2
```

## Run the Application

Start Ollama and run:

```cmd
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Workflow

1. Enter the sender email and email content.
2. Analyze and classify the email.
3. Generate a draft reply.
4. Review and edit the draft.
5. Approve or reject the response.

## Current Limitations

- The application generates drafts but does not send emails automatically.
- Workflow state uses an in-memory checkpointer.
- State is not preserved after restarting the application.

## Future Enhancements

- Gmail or Microsoft Graph integration.
- Persistent database storage.
- Automated tests and monitoring.
- Secure authentication and deployment.

## Security

Do not commit passwords, API keys, or access tokens to GitHub.
