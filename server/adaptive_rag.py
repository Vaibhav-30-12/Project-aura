from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
from langchain_community.document_loaders import WebBaseLoader#type: ignore
from langchain_community.vectorstores import Chroma#type: ignore
from langchain_openai import OpenAIEmbeddings#type: ignore
from langchain_core.prompts import ChatPromptTemplate#type: ignore
from langchain_openai import ChatOpenAI#type: ignore
from pydantic import BaseModel, Field#type: ignore
from typing import Literal, List
from typing_extensions import TypedDict
from langchain.schema import Document#type: ignore
from langchain import hub#type: ignore
from langchain_core.output_parsers import StrOutputParser#type: ignore
from langchain_community.tools.tavily_search import TavilySearchResults#type: ignore
from langgraph.graph import END, StateGraph, START#type: ignore
from pprint import pprint
import os
from dotenv import load_dotenv#type: ignore
from langchain.tools import StructuredTool
from llama_index.retrievers.pathway import PathwayRetriever
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PathwayVectorClient
load_dotenv()