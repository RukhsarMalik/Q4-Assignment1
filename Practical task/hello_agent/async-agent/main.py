import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Setup the client to connect to Gemini OpenAI-compatible endpoint
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define your model specifying the Gemini model name
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# RunConfig for additional options like tracing, timeouts, etc.
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Async main function to create agent and run prompt
async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model=model,
    )

    # Run prompt asynchronously and print result
    result = await Runner.run(agent, "Tell me about Pakistan in 2 lines", run_config=config)
    print(result.final_output)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
