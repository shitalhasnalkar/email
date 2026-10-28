
import uuid
from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


# =========================================================
# 1. EMAIL CLASSIFICATION SCHEMA
# =========================================================

class EmailClassification(BaseModel):
    intent: Literal[
        "question", "bug", "billing", "feature", "complex"
    ] = Field(description="Main purpose of the email")

    urgency: Literal[
        "low", "medium", "high", "critical"
    ] = Field(description="Urgency of the email")

    topic: str = Field(description="Short topic of the email")
    summary: str = Field(description="Brief summary of the email")


# =========================================================
# 2. STATE SHARED BETWEEN LANGGRAPH NODES
# =========================================================

class EmailState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    classification: dict
    ticket_id: str
    search_results: str
    customer_history: str
    draft_response: str
    approval_decision: str
    final_status: str


# =========================================================
# 3. LOCAL OLLAMA MODEL
# =========================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Convert model output into validated Pydantic fields.
structured_llm = llm.with_structured_output(EmailClassification)


# =========================================================
# 4. MOCK CUSTOMER DATA AND DOCUMENTATION
# =========================================================

CUSTOMER_HISTORY = {
    "customer@example.com": (
        "Customer has contacted support previously "
        "about subscription and payment questions."
    )
}

KNOWLEDGE_BASE = {
    "billing": (
        "Billing support: Check the invoice and payment history. "
        "If a duplicate charge is suspected, verify both transactions "
        "before arranging a refund."
    ),
    "login": (
        "Login support: Check the registered email address, "
        "reset the password, and verify account access."
    ),
    "bug": (
        "Technical support: Collect reproduction steps, "
        "the error message, and relevant system information."
    ),
    "subscription": (
        "Subscription support: Review the current plan, renewal date, "
        "and payment records before changing the subscription."
    ),
    "feature": (
        "Feature requests: Record the requested functionality, "
        "the use case, and the expected benefit."
    ),
}


# =========================================================
# 5. NODE: READ EMAIL
# =========================================================

def read_email(state: EmailState):
    print("\n[1] Reading email...")

    history = CUSTOMER_HISTORY.get(
        state["sender_email"],
        "No previous customer history found."
    )

    print("Sender:", state["sender_email"])
    print("Email ID:", state["email_id"])

    return {
        "customer_history": history,
        "final_status": "processing",
    }


# =========================================================
# 6. NODE: CLASSIFY EMAIL
# =========================================================

def classify_intent(state: EmailState):
    print("\n[2] Classifying email with Ollama...")

    prompt = f"""
You are an email support classification assistant.

Classify the following customer email.

Rules:
- question: a general information question
- bug: a software error or technical malfunction
- billing: a payment, invoice, refund, or charge issue
- feature: a request for a new feature
- complex: an issue requiring several steps or specialist review

Urgency:
- low: general question with no immediate impact
- medium: normal support issue
- high: significant customer impact
- critical: severe outage, security incident, or urgent major impact

Do not invent facts. Use the email content.

Customer email:
{state["email_content"]}
"""

    result = structured_llm.invoke(prompt)
    classification = result.model_dump()

    print("Classification:", classification)

    return {"classification": classification}


# =========================================================
# 7. NODE: SEARCH DOCUMENTATION
# =========================================================

def search_documentation(state: EmailState):
    print("\n[3] Searching support documentation...")

    classification = state["classification"]
    text = (
        state["email_content"] + " " +
        classification["topic"] + " " +
        classification["summary"]
    ).lower()

    matched = []

    # Match documentation using simple keywords.
    for keyword, document in KNOWLEDGE_BASE.items():
        if keyword in text:
            matched.append(document)

    if not matched:
        matched.append(
            "No directly matching article was found. "
            "Ask for additional information or escalate the issue."
        )

    results = "\n".join(f"- {item}" for item in matched)

    print(results)

    return {"search_results": results}


# =========================================================
# 8. NODE: CREATE A MOCK SUPPORT TICKET
# =========================================================

def bug_tracking(state: EmailState):
    print("\n[4] Checking ticket requirements...")

    classification = state["classification"]

    if classification["intent"] in ("bug", "billing", "complex"):
        ticket_id = "TICKET-" + uuid.uuid4().hex[:8].upper()
        print("Mock ticket created:", ticket_id)
    else:
        ticket_id = "N/A"
        print("No ticket required for this email.")

    return {"ticket_id": ticket_id}


# =========================================================
# 9. NODE: DRAFT RESPONSE
# =========================================================

def write_response(state: EmailState):
    print("\n[5] Drafting response with Ollama...")

    classification = state["classification"]

    prompt = f"""
You are a professional customer support agent.

Write a helpful, polite, concise email response.

Customer email:
{state["email_content"]}

Classification:
{classification}

Customer history:
{state["customer_history"]}

Relevant support documentation:
{state["search_results"]}

Ticket ID:
{state["ticket_id"]}

Rules:
- Address the customer's actual concern.
- Use only facts provided in the email and documentation.
- Do not claim a refund has been processed.
- Do not claim a bug has been fixed.
- Do not invent policies, timelines, or actions.
- If information is missing, ask a clear follow-up question.
- If a ticket ID exists, mention it where appropriate.
- Return only the email body, without commentary.
"""

    response = llm.invoke(prompt)
    draft = response.content

    if not isinstance(draft, str):
        draft = str(draft)

    print("\nDraft response:\n")
    print(draft)

    return {"draft_response": draft}


# =========================================================
# 10. CONDITIONAL ROUTING
# =========================================================

def route_after_drafting(state: EmailState) -> str:
    classification = state["classification"]

    if (
        classification["urgency"] in ("high", "critical")
        or classification["intent"] == "complex"
    ):
        return "human_review"

    return "send_reply"


# =========================================================
# 11. NODE: HUMAN APPROVAL
# =========================================================

def human_review(state: EmailState):
    print("\n[6] Human approval required.")

    # LangGraph pauses here until the workflow is resumed.
    decision = interrupt({
        "message": "Review the email draft before sending.",
        "sender": state["sender_email"],
        "classification": state["classification"],
        "draft_response": state["draft_response"],
        "ticket_id": state["ticket_id"],
        "allowed_actions": ["approve", "edit", "reject"],
    })

    action = decision.get("action", "reject").lower()

    if action == "approve":
        return {
            "approval_decision": "approved",
            "final_status": "approved",
        }

    if action == "edit":
        edited_draft = decision.get(
            "draft_response", state["draft_response"]
        )
        return {
            "draft_response": edited_draft,
            "approval_decision": "approved",
            "final_status": "approved",
        }

    return {
        "approval_decision": "rejected",
        "final_status": "rejected",
    }


# =========================================================
# 12. NODE: SIMULATED EMAIL SENDING
# =========================================================

def send_reply(state: EmailState):
    print("\n[7] Preparing email delivery...")

    # This is a demonstration only. No email is actually sent.
    print("To:", state["sender_email"])
    print("Subject: Re: Your support request")
    print("\nEmail body:\n")
    print(state["draft_response"])

    return {"final_status": "sent_demo"}


# =========================================================
# 13. BUILD THE LANGGRAPH WORKFLOW
# =========================================================

builder = StateGraph(EmailState)

builder.add_node("read_email", read_email)
builder.add_node("classify_intent", classify_intent)
builder.add_node("search_documentation", search_documentation)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("write_response", write_response)
builder.add_node("human_review", human_review)
builder.add_node("send_reply", send_reply)

builder.add_edge(START, "read_email")
builder.add_edge("read_email", "classify_intent")
builder.add_edge("classify_intent", "search_documentation")
builder.add_edge("search_documentation", "bug_tracking")
builder.add_edge("bug_tracking", "write_response")

builder.add_conditional_edges(
    "write_response",
    route_after_drafting,
    {
        "human_review": "human_review",
        "send_reply": "send_reply",
    },
)

builder.add_conditional_edges(
    "human_review",
    lambda state: (
        "send_reply"
        if state["approval_decision"] == "approved"
        else END
    ),
    {
        "send_reply": "send_reply",
        END: END,
    },
)

builder.add_edge("send_reply", END)

# In-memory checkpointing supports pause and resume
# while this Python process remains running.
checkpointer = InMemorySaver()

app = builder.compile(checkpointer=checkpointer)


# =========================================================
# 14. RUN THE EMAIL AGENT
# =========================================================

if __name__ == "__main__":

    initial_state: EmailState = {
        "email_content": (
            "Hello Support Team, I was charged twice for my "
            "subscription this month. Could you please check "
            "the duplicate payment and advise me?"
        ),
        "sender_email": "customer@example.com",
        "email_id": "EMAIL-001",
        "classification": {},
        "ticket_id": "",
        "search_results": "",
        "customer_history": "",
        "draft_response": "",
        "approval_decision": "",
        "final_status": "",
    }

    config = {
        "configurable": {
            "thread_id": "email-demo-001"
        }
    }

    print("=" * 60)
    print("LANGGRAPH EMAIL AGENT - OLLAMA")
    print("=" * 60)

    result = app.invoke(initial_state, config=config)

    # Handle human-in-the-loop approval.
    if result.get("__interrupt__"):

        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 60)
        print("HUMAN REVIEW")
        print("=" * 60)
        print("Sender:", interrupt_data["sender"])
        print("Classification:", interrupt_data["classification"])
        print("Ticket:", interrupt_data["ticket_id"])
        print("\nDraft:\n", interrupt_data["draft_response"])

        while True:
            action = input(
                "\nChoose approve, edit, or reject: "
            ).strip().lower()

            if action in ("approve", "edit", "reject"):
                break

            print("Please enter approve, edit, or reject.")

        resume_data = {"action": action}

        if action == "edit":
            resume_data["draft_response"] = input(
                "\nEnter the revised email body:\n"
            )

        result = app.invoke(
            Command(resume=resume_data),
            config=config,
        )

    print("\n" + "=" * 60)
    print("WORKFLOW FINISHED")
    print("=" * 60)
    print("Final status:", result.get("final_status"))
    print("Ticket ID:", result.get("ticket_id"))
