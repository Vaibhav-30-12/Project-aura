from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
import json
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"


# --------------------------------------------HELPER FUNCTIONS-------------------------------------------------------------------------

# Load persona JSON file
def load_persona_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
