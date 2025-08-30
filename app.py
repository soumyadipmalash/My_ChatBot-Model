from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# Load environment variables
load_dotenv(override=True)

def push(text):
    """Send notification via Pushover (optional)"""
    if os.getenv("PUSHOVER_TOKEN") and os.getenv("PUSHOVER_USER"):
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            }
        )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

# Tools for structured responses
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if provided"},
            "notes": {"type": "string", "description": "Additional conversation details"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unanswered question"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]

class Me:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key missing! Set OPENROUTER_API_KEY or OPENAI_API_KEY in Hugging Face Secrets.")

        # Use OpenRouter API if key exists
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

        self.openai = OpenAI(api_key=api_key, base_url=base_url)
        self.name = "Soumyadip Malash"

        # Load resume text
        self.resume = ""
        if os.path.exists("me/Resume_Soumyadip Malash_FinalTillDate.pdf"):
            try:
                reader = PdfReader("me/Resume_Soumyadip Malash_FinalTillDate.pdf")
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        self.resume += text
            except Exception:
                self.resume = "Could not load resume text."

        # Load summary text
        self.summary = "No summary available."
        if os.path.exists("me/summary.txt"):
            try:
                with open("me/summary.txt", "r", encoding="utf-8") as f:
                    self.summary = f.read()
            except Exception:
                pass

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        return (
            f"You are acting as {self.name}. Answer questions about {self.name}'s career, background, and skills. "
            f"Be professional and engaging, as if talking to a potential employer. "
            f"If you don't know something, use the record_unknown_question tool. "
            f"Encourage users to share their email and use record_user_details tool. "
            f"\n\n## Summary:\n{self.summary}\n\n## Resume:\n{self.resume}\n\n"
        )

    def chat(self, message, history):
        try:
            messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
            done = False
            while not done:
                response = self.openai.chat.completions.create(
                    model="deepseek/deepseek-r1", 
                    messages=messages,
                    tools=tools,
                    max_tokens=3679,  
                    temperature=0.7
                )
                if response.choices[0].finish_reason == "tool_calls":
                    message = response.choices[0].message
                    tool_calls = message.tool_calls
                    results = self.handle_tool_call(tool_calls)
                    messages.append(message)
                    messages.extend(results)
                else:
                    done = True
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return f"Error occurred: {e}. Possible cause: insufficient credits or API issue."

# Launch Gradio app
if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
