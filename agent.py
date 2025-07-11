import os
import asyncio
import uuid
import gradio as gr
from google.adk import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Key Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. "
                     "Please set it in your .env file or system environment. "
                     "Get your API key from Google AI Studio (aistudio.google.com/app/apikey).")
genai.configure(api_key=api_key)

# --- App + Session Setup ---
APP_NAME = "LostFoundApp"
USER_ID = "visitor"
SESSION_ID = str(uuid.uuid4())
session_service = InMemorySessionService()

runner = None
root_agent = None

# --- Setup function ---
async def setup():
    global runner, root_agent

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"user_goal": "Learn about the Lost & Found platform"}
    )

    root_agent = Agent(
        name="LostFoundHelper",
        model="gemini-2.5-pro",  # Changed to flash for better quota management
        instruction="""
You are a helpful, intelligent, and friendly AI assistant.

Your primary goal is to answer user questions comprehensively and accurately.

You have specific, detailed knowledge about the 'Lostloop Web Application'. When a user asks about it, prioritize this internal knowledge.

📄 Application Overview:
The Lostloop Web Application is a user-friendly platform designed to help people report and recover lost items efficiently.

👤 Account System:
- To create an account, click on Sign Up, then enter your email and password.
- To log in, use the same email and password on the login screen.
- To log out, click on your profile or logout button to safely exit.

Creating an account allows you to track your reports, update them later, and access additional features like chat history.

📌 Core Features:
✅ Submit Report — Click 'Report', fill in the item details, upload a photo if needed, and submit.
✅ Edit Details — Go to the item page and click 'Edit' to change any information.
✅ Delete Report — Use the delete option to permanently remove a report.
✅ Google Maps Integration — The item's location is shown on an interactive map.
✅ QR Code — Each item has a QR code that can be scanned to open the item's page.
✅ Chat Option — Open the item page and use the chat feature to message the reporter or finder.
✅ Search — Use filters like type, status, location, or keywords to find items.
✅ Claim/Unclaim — Toggle the item's status to mark it as claimed or unclaimed.
✅ View Image — Images uploaded with the item can be viewed on the item detail page.
✅ Contact Reporter — Use the email shown on the item page to directly contact the item owner.
✅ Contact Lostloop — Go to the Support page to send queries or issues to the Lostloop team.

🌟 Who Is It For?
- Campuses, offices, events, and public spaces — anywhere people may lose or find belongings.

📞 Need Help?
- Still confused? Ask your question here or visit the Support section inside the app.

You can also search the web if the question is unrelated to Lostloop. Be warm, concise, and helpful in all responses.
""",
        description="An intelligent, friendly AI assistant that can answer general questions using Google Search and also has specific knowledge about the Lost & Found web application.",
        tools=[google_search]
    )

    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

# --- Chat function ---
async def chat(user_message, history):
    if runner is None:
        history.append({"role": "assistant", "content": "⚠ Chatbot is still initializing. Please wait and try again."})
        yield history, ""
        return

    history.append({"role": "user", "content": user_message})
    new_message_for_runner = types.Content(role="user", parts=[types.Part(text=user_message)])
    bot_reply_text = ""

    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=new_message_for_runner):
            if event.is_final_response() and event.content and event.content.parts:
                bot_reply_text = event.content.parts[0].text
                history.append({"role": "assistant", "content": bot_reply_text})
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠ Error: {str(e)}"})

    yield history, "" # Yield the final state of history and clear input

# --- Clear chat history ---
def clear_history():
    return [], ""

# --- Main app ---
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(setup())

    with gr.Blocks(theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate")) as demo:
        gr.Markdown("<h1 style='color:black;text-align:center;'>🔍 Lost & Found Agent</h1>", elem_id="title")
        chatbot = gr.Chatbot([], elem_id="chatbot", height=500, type='messages')
        msg = gr.Textbox(label="Your Message", placeholder="Type your question here…")
        clear = gr.Button("Clear")
        state = gr.State([])

        msg.submit(chat, [msg, state], [chatbot, msg])
        clear.click(clear_history, None, [chatbot, msg])

    demo.launch(share=True,debug=True)
