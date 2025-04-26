# ----------------------------------------
# 📄 File: main.py
# 📚 Purpose: AI Research Agent (LangChain, OpenAI, Anthropic Ready)
# ----------------------------------------

# --- Imports ---
from dotenv import load_dotenv  # Load environment variables
from pydantic import BaseModel   # Define structured outputs
from langchain_openai import ChatOpenAI  # OpenAI LLM
from langchain_anthropic import ChatAnthropic  # (Optional) Anthropic LLM
from langchain_core.prompts import ChatPromptTemplate  # Create prompt templates
from langchain_core.output_parsers import PydanticOutputParser  # Parse LLM outputs into Pydantic models
from langchain.agents import create_tool_calling_agent, AgentExecutor  # Create tool-using agents
from tools import search_tool, wiki_tool, save_tool  # Custom tools

# --- Environment Setup ---
load_dotenv()  # Load environment variables from .env file (e.g., API keys)

# --- Define Structured Output Schema ---
class ResearchResponse(BaseModel):
    """Defines the expected structure for research responses."""
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# --- Initialize LLMs ---
# ✅ Primary: OpenAI (currently rate-limited, handle with care)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 🔄 Optional Backup: Anthropic Claude (future integration)
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# --- Output Parser Setup ---
# Tells the LLM how to format its output according to ResearchResponse
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# --- Prompt Template Setup ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         You are a research assistant tasked with generating structured research papers.
         Use necessary tools to gather information.
         Wrap the final output using the below format strictly:\n{format_instructions}
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# --- Tool Setup ---
# Defines external tools available to the agent
tools = [search_tool, wiki_tool, save_tool]

# --- Create the Tool-Enabled Agent ---
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

# --- Setup the Agent Executor ---
# Wrap the agent in an executor that manages execution
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Logs every step for easier debugging
)

# --- User Interaction ---
# Collect user query
query = input("What can I help you research? ")

# --- Agent Execution ---
# Agent processes the query using available tools
raw_response = agent_executor.invoke({"query": query})

# --- Output Parsing ---
# Attempt to parse and display the structured research response
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("🚨 Error parsing response:", e)
    print("📋 Raw Response for Debugging:", raw_response)
