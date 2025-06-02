import os
from dotenv import load_dotenv
import chainlit as cl
import litellm

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@cl.on_message
async def main(message: cl.Message):
    try:
        response = litellm.completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "user", "content": message.content}
            ],
            api_key=GROQ_API_KEY   # Pass the key here
        )
        await cl.Message(content=response['choices'][0]['message']['content']).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()
