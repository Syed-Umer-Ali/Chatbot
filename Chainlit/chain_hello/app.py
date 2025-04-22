import chainlit as cl
import os

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Set up the provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Set up the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# Step 3: Configure the agent: Define at run level
runconfig = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Step 4: Define the agent
agent1 = Agent(
    instructions="You are a helpful assistant that can answer questions and help with tasks.",
    name="First Agent",
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm the first agent. How can I help you today?").send()

# final step: run the agent

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
         agent1,
        input = message.content,
        run_config=runconfig,
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()

