import json
from mistralai.client import Mistral
from dotenv import load_dotenv
from rag import compare, retrieve
from subjects import DOMAINS
import os

load_dotenv()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": "Compare two subjects in depth using the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_a": {
                        "type": "string"
                    },
                    "subject_b": {
                        "type": "string"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional: football, movies, or music"
                    },
                },
                "required": ["subject_a", "subject_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Fetch knowledge base context about a single subject.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional: football, movies, or music"
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_domains",
            "description": "List available domains in the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    if name == "compare":
        return compare(**args)
    if name == "retrieve":
        matches = retrieve(**args)
        if not matches:
            return "No results found."
        return "\n\n".join(f"[{m['name']}]\n{m['text']}" for m in matches)
    if name == "list_domains":
        return f"Available domains: {', '.join(DOMAINS)}"
    return f"Unknown tool: {name}"


def chat(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to a knowledge base. "
                "Only answer based on what the tools return. "
                "If the user's question contains a false premise or the tool result does not support the answer, "
                "correct the premise and do not answer the question as asked. "
                "Never guess or hallucinate facts. "
                "At the end of every response, sign it with (ANAS CHERIET)."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat.complete(model=MISTRAL_MODEL, tools=TOOLS, messages=messages)
    msg = response.choices[0].message

    if not msg.tool_calls:
        print("[agent] no tool call — answered directly")
        return msg.content

    for call in msg.tool_calls:
        print(f"[agent] tool selected: {call.function.name}")
        print(f"[agent] arguments: {call.function.arguments}")

    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = run_tool(call.function.name, args)
        print(f"[agent] tool result preview: {result[:100]}...")
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

    print("[agent] generating final response...")
    final = client.chat.complete(model=MISTRAL_MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    print("AgnosticLookup Agent — type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        answer = chat(user_input)
        print(f"\nAgent: {answer}\n")
