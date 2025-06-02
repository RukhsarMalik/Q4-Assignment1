import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()

# Get GEMINI_API_KEY from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

# Set up AsyncOpenAI client (Gemini-compatible OpenAI wrapper)
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create the Gemini model using OpenAI-compatible interface
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)


config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Create an Agent with instructions
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model,
)

# Run the agent synchronously with a prompt
print("\nCALLING AGENT...\n")
result = Runner.run_sync(agent, "Hello, how are you?", run_config=config)

#  Print final output
print("Agent Response:\n")
print(result.final_output)
