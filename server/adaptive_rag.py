# Qdrant-enabled RAG pipeline with LangGraph

from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import WebBaseLoader  # type: ignore
from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from typing import Literal
from langchain.schema import Document  # type: ignore
from langchain import hub  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langgraph.graph import END, StateGraph, START  # type: ignore
from pprint import pprint
import os
from dotenv import load_dotenv  # type: ignore
from langchain.vectorstores import Qdrant  # <-- Updated to use Qdrant
from qdrant_client import QdrantClient  # <-- Qdrant client
from qdrant_client.models import Distance, VectorParams  # <-- Qdrant config

# ===================== Load environment variables =====================
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# ===================== Qdrant Setup =====================
# This replaces Pathway client setup
client = QdrantClient(url="http://localhost:6333")  # or Qdrant Cloud endpoint
collection_name = "my_rag_chunks"

# (Re)create the collection (index)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Set up OpenAI Embeddings
embd = OpenAIEmbeddings()

# Vectorstore using Qdrant
vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embd,
)

# Add documents (only needed once — skip on reruns)
docs = [
    Document(page_content="Qdrant is a vector database."),
    Document(page_content="LangChain helps build LLM apps."),
    Document(page_content="This is a personal project using RAG."),
]
vectorstore.add_documents(docs)

# Create retriever from Qdrant vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # <-- Updated retriever

# ===================== Question Router =====================
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(...)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to SEC fillings or financial data of multiple companies.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_router),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# ===================== Document Grader =====================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)
system_grader = """You are a grader assessing relevance of a retrieved document to a user question.\n \
1)If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant\n\n2)It is OK if the facts have SOME information that is unrelated to the question (1) is met\n\nGive a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grade_prompt | structured_llm_doc_grader

# ===================== RAG Chain =====================
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm | StrOutputParser()

# ===================== Hallucination Grader =====================
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeHallucinations)
system_hallucination_grader = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n \
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_hallucination_grader),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])
hallucination_grader = hallucination_prompt | structured_llm_grader
