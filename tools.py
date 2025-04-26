# ----------------------------------------
# 📄 File: tools.py
# 🛠 Purpose: Custom Tools for the AI Research Agent
# ----------------------------------------

# --- Imports ---
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# --- Save Tool ---
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """
    Saves structured research data to a local text file with a timestamp.

    Args:
        data (str): The research data to save.
        filename (str, optional): Filename to save the data. Defaults to 'research_output.txt'.

    Returns:
        str: Success message with filename.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = (
        f"--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )

    # Append the data to the file
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"✅ Data successfully saved to '{filename}'."

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Persists structured research data into a local text file."
)

# --- Search Tool (DuckDuckGo) ---
search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",
    func=search.run,
    description="🌐 Perform a web search for real-time information retrieval."
)

# --- Wikipedia Tool ---
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,                # Limit to 1 best result
    doc_content_chars_max=100        # Restrict content length for quick lookups
)

wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
# 📚 Direct Wikipedia querying tool, optimized for concise results
