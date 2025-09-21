# # # pdf_to_chroma.py

# # import os
# # from langchain_community.document_loaders import PyMuPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.embeddings import OpenAIEmbeddings  # or HuggingFaceEmbeddings
# # from langchain_community.vectorstores import Chroma

# # # --- Set your OpenAI key ---
# # os.environ["OPENAI_API_KEY"] = "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"  # replace with your key

# # # --- Load PDF files from a folder ---
# # pdf_dir = "pdfs"  # folder containing your PDFs
# # all_docs = []

# # for filename in os.listdir(pdf_dir):
# #     if filename.endswith(".pdf"):
# #         loader = PyMuPDFLoader(os.path.join(pdf_dir, filename))
# #         docs = loader.load()
# #         all_docs.extend(docs)

# # print(f"✅ Loaded {len(all_docs)} documents from PDFs.")

# # # --- Split text into chunks ---
# # splitter = RecursiveCharacterTextSplitter(
# #     chunk_size=1000,
# #     chunk_overlap=200
# # )
# # chunks = splitter.split_documents(all_docs)

# # print(f"🧩 Split into {len(chunks)} text chunks.")

# # # --- Embeddings model ---
# # embeddings = OpenAIEmbeddings()  # or HuggingFaceEmbeddings()

# # # --- Store into ChromaDB ---
# # vector_db = Chroma.from_documents(
# #     documents=chunks,
# #     embedding=embeddings,
# #     persist_directory="chroma_pdf_db"
# # )

# # vector_db.persist()
# # print("✅ Embeddings stored in Chroma vector DB.")


# from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
# from langchain_community.document_loaders import WebBaseLoader  # type: ignore
# from langchain_openai import OpenAIEmbeddings  # type: ignore
# from langchain_core.prompts import ChatPromptTemplate  # type: ignore
# from langchain_openai import ChatOpenAI  # type: ignore
# from pydantic import BaseModel, Field  # type: ignore
# from typing import Literal, List
# from typing_extensions import TypedDict
# from langchain.schema import Document  # type: ignore
# from langchain import hub  # type: ignore
# from langchain_core.output_parsers import StrOutputParser  # type: ignore
# from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
# from langgraph.graph import END, StateGraph, START  # type: ignore
# from pprint import pprint
# import os
# from dotenv import load_dotenv  # type: ignore
# from langchain.tools import StructuredTool
# from langchain_community.vectorstores import Qdrant  # ✅ NEW Qdrant
# from qdrant_client import QdrantClient  # ✅ Qdrant client
# from qdrant_client.http import models  # ✅ Qdrant models

# load_dotenv()

# os.environ['OPENAI_API_KEY'] = "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"

# # Initialize LLM and embeddings
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# embeddings = OpenAIEmbeddings()

# # Initialize Qdrant client
# # qdrant_client = QdrantClient(
# #     path="./qdrant_db",  # Local storage path
# #     # For cloud/remote Qdrant, use:
# #     # url="http://localhost:6333",  # or your Qdrant server URL
# #     # api_key="your-api-key"  # if authentication is required
# # )
# qdrant_client = QdrantClient(
#     url="https://b46a2c8f-2e88-4883-adc7-50ea89e775e3.us-west-1-0.aws.cloud.qdrant.io:6333", 
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzUxMTQzNjgwfQ.s54bLE4-tQn0C4t-nP-tMggZl5RJsk3MlmfCjdbLfow",
# )

# # Collection name
# COLLECTION_NAME = "pdf_documents"

# # Initialize Qdrant vector store
# try:
#     vector_store = Qdrant(
#         client=qdrant_client,
#         collection_name=COLLECTION_NAME,
#         embeddings=embeddings,
#     )
#     print(f"Connected to Qdrant collection: {COLLECTION_NAME}")
# except Exception as e:
#     print(f"Error connecting to Qdrant: {e}")
#     # Create collection if it doesn't exist
#     try:
#         qdrant_client.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=models.VectorParams(
#                 size=1536,  # OpenAI embedding dimension
#                 distance=models.Distance.COSINE
#             )
#         )
#         vector_store = Qdrant(
#             client=qdrant_client,
#             collection_name=COLLECTION_NAME,
#             embeddings=embeddings,
#         )
#         print(f"Created and connected to Qdrant collection: {COLLECTION_NAME}")
#     except Exception as create_error:
#         print(f"Error creating collection: {create_error}")

# # Create retriever from vector store
# retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# # Initialize client reference for direct operations
# client = vector_store

# #============================= QUESTION ROUTER =================================
# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant datasource."""

#     datasource: Literal["vectorstore", "web_search"] = Field(
#         ...,
#         description="Given a user question choose to route it to web search or a vectorstore.",
#     )

# structured_llm_router = llm.with_structured_output(RouteQuery)

# # Prompt
# system_router = """You are an expert at routing a user question to a vectorstore or web search.
# The vectorstore contains documents related to SEC fillings or financial data of multiple companies.
# Use the vectorstore for questions on these topics. Otherwise, use web-search."""
# route_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_router),
#         ("human", "{question}"),
#     ]
# )

# question_router = route_prompt | structured_llm_router

# #================================================================================

# #============================= GRADE DOCUMENTS =================================
# # Data model
# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""

#     binary_score: str = Field(
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )

# # LLM with function call
# structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)

# # Prompt
# system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
#     1)If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant\n
#     2)It is OK if the facts have SOME information that is unrelated to the question (1) is met \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
# grade_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_grader),
#         ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#     ]
# )

# retrieval_grader = grade_prompt | structured_llm_doc_grader

# #================================================================================

# #==============================RAG CHAIN=========================================
# # Prompt
# prompt = hub.pull("rlm/rag-prompt")

# # LLM
# llm_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# # Post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Chain
# rag_chain = prompt | llm_mini | StrOutputParser()

# #============================= GRADE HALLUCINATIONS =================================
# # Data model
# class GradeHallucinations(BaseModel):
#     """Binary score for hallucination present in generation answer."""

#     binary_score: str = Field(
#         description="Answer is grounded in the facts, 'yes' or 'no'"
#     )

# # LLM with function call
# structured_llm_grader = llm_mini.with_structured_output(GradeHallucinations)

# # Prompt
# system_hallucination_grader = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
#      Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
# hallucination_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_hallucination_grader),
#         ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
#     ]
# )

# hallucination_grader = hallucination_prompt | structured_llm_grader
# #==================================================================================

# #============================= GRADE ANSWER =======================================
# # Data model
# class GradeAnswer(BaseModel):
#     """Binary score to assess answer addresses question."""

#     binary_score: str = Field(
#         description="Answer addresses the question, 'yes' or 'no'"
#     )

# # LLM with function call
# structured_llm_answer_grader = llm_mini.with_structured_output(GradeAnswer)

# # Prompt
# system_answer_grader = """You are a grader assessing whether an answer addresses / resolves a question \n 
#      Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
# answer_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_answer_grader),
#         ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
#     ]
# )

# answer_grader = answer_prompt | structured_llm_answer_grader
# #==================================================================================

# #============================= QUERY REWRITER ====================================
# # Prompt
# system_rewriter = """
# You are a question re-writer that converts an input question to a better version that is optimized \n 
# for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
# And reformulate the query to better suite the content of the documents in the vectorstore, which are mainly SEC filings and financial documents."""
# re_write_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_rewriter),
#         (
#             "human",
#             "Here is the initial question: \n\n {question} \n Formulate an improved question.",
#         ),
#     ]
# )

# question_rewriter = re_write_prompt | llm | StrOutputParser()

# #==================================================================================

# #============================= WEB SEARCH ========================================
# web_search_tool = TavilySearchResults(k=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# #==================================================================================

# #============================= WORKFLOW ===========================================
# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.

#     Attributes:
#         question: question
#         generation: LLM generation
#         documents: list of documents
#         count: Number of times retriever is called
#         queries: list of possible queries
#     """

#     question: str
#     generation: str
#     documents: List[str]
#     count: int
#     queries: List[str]
#     company_name: str
#     year: str
#     table: str
#     mode: str

# from constants import get_object

# def retrieve(state):
#     """
#     Retrieve documents using Qdrant

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     print("---RETRIEVE---")
#     question = state["question"]
#     queries = state["queries"]
#     count = state["count"] + 1
#     print('======STATE BEFORE RETRIEVAL==========')
    
#     try:
#         get_object().emit("update", {
#             "username": "Retriever",
#             "isAgent": False,
#             "parentAgent": "Research",
#             "content": "Retrieving documents from Qdrant...",
#             "isUser": False,
#             "verdict": "passing to next agent",
#         })
#     except Exception as e:
#         pass
    
#     print(state)
    
#     # Retrieval using Qdrant
#     documents = []
#     all_results = []
    
#     company_name = state.get('company_name', '')
#     year = state.get('year', '')
    
#     # Build Qdrant filter
#     qdrant_filter = None
#     if company_name or year:
#         conditions = []
#         if company_name:
#             conditions.append(
#                 models.FieldCondition(
#                     key="company_name",
#                     match=models.MatchText(text=company_name.lower())
#                 )
#             )
#         if year:
#             conditions.append(
#                 models.FieldCondition(
#                     key="year",
#                     match=models.MatchText(text=year)
#                 )
#             )
        
#         if conditions:
#             qdrant_filter = models.Filter(
#                 must=conditions
#             )
    
#     if queries[0] != "" and count == 1:
#         # Use multiple queries
#         for query in queries:
#             # Search for table content
#             table_query = f"Markdown Table {query}"
#             try:
#                 table_docs = vector_store.similarity_search_with_score(
#                     table_query, 
#                     k=10,
#                     filter=qdrant_filter
#                 )
#                 for doc, score in table_docs:
#                     if should_include_document(doc, company_name, year):
#                         all_results.append((doc, score, 'table'))
#             except Exception as e:
#                 print(f"Table search error: {e}")
#                 # Fallback without filter
#                 try:
#                     table_docs = vector_store.similarity_search_with_score(table_query, k=10)
#                     for doc, score in table_docs:
#                         if should_include_document(doc, company_name, year):
#                             all_results.append((doc, score, 'table'))
#                 except Exception as e2:
#                     print(f"Table search fallback error: {e2}")
            
#             # Search for regular text content
#             try:
#                 text_docs = vector_store.similarity_search_with_score(
#                     query, 
#                     k=10,
#                     filter=qdrant_filter
#                 )
#                 for doc, score in text_docs:
#                     if should_include_document(doc, company_name, year):
#                         all_results.append((doc, score, 'text'))
#             except Exception as e:
#                 print(f"Text search error: {e}")
#                 # Fallback without filter
#                 try:
#                     text_docs = vector_store.similarity_search_with_score(query, k=10)
#                     for doc, score in text_docs:
#                         if should_include_document(doc, company_name, year):
#                             all_results.append((doc, score, 'text'))
#                 except Exception as e2:
#                     print(f"Text search fallback error: {e2}")
#     else:
#         # Single query search
#         table_query = f"Markdown Table {question}"
#         try:
#             table_results = vector_store.similarity_search_with_score(
#                 table_query, 
#                 k=10,
#                 filter=qdrant_filter
#             )
#             for doc, score in table_results:
#                 if should_include_document(doc, company_name, year):
#                     all_results.append((doc, score, 'table'))
#         except Exception as e:
#             print(f"Table search error: {e}")
#             try:
#                 table_results = vector_store.similarity_search_with_score(table_query, k=10)
#                 for doc, score in table_results:
#                     if should_include_document(doc, company_name, year):
#                         all_results.append((doc, score, 'table'))
#             except Exception as e2:
#                 print(f"Table search fallback error: {e2}")
        
#         try:
#             text_results = vector_store.similarity_search_with_score(
#                 question, 
#                 k=10,
#                 filter=qdrant_filter
#             )
#             for doc, score in text_results:
#                 if should_include_document(doc, company_name, year):
#                     all_results.append((doc, score, 'text'))
#         except Exception as e:
#             print(f"Text search error: {e}")
#             try:
#                 text_results = vector_store.similarity_search_with_score(question, k=10)
#                 for doc, score in text_results:
#                     if should_include_document(doc, company_name, year):
#                         all_results.append((doc, score, 'text'))
#             except Exception as e2:
#                 print(f"Text search fallback error: {e2}")

#     # Remove duplicates based on page_content
#     unique_results = []
#     seen_content = set()
#     for doc, score, doc_type in all_results:
#         content_hash = hash(doc.page_content)
#         if content_hash not in seen_content:
#             unique_results.append((doc, score, doc_type))
#             seen_content.add(content_hash)
    
#     # Sort by similarity score (higher is better for Qdrant cosine similarity)
#     unique_results.sort(key=lambda x: x[1], reverse=True)
    
#     # Prioritize table results, then text results
#     table_results = [r for r in unique_results if r[2] == 'table'][:3]
#     text_results = [r for r in unique_results if r[2] == 'text'][:5]
    
#     # Combine results
#     final_results = table_results + text_results
#     documents = [doc.page_content for doc, score, doc_type in final_results[:10]]
    
#     print(f"Retrieved {len(documents)} documents")
#     if len(documents) == 0:
#         print("No documents found, trying broader search...")
#         # Try a broader search without filters
#         try:
#             broad_results = vector_store.similarity_search_with_score(question, k=20)
#             documents = [doc.page_content for doc, score in broad_results[:10]]
#             print(f"Broad search retrieved {len(documents)} documents")
#         except Exception as e:
#             print(f"Broad search error: {e}")
    
#     return {"documents": documents, "question": question, "count": count}

# def should_include_document(doc, company_name, year):
#     """
#     Check if document should be included based on company name and year
#     """
#     if not company_name and not year:
#         return True
    
#     # Check in metadata first
#     metadata = getattr(doc, 'metadata', {})
    
#     # Check various metadata fields
#     metadata_fields = ['source', 'path', 'file_name', 'company_name', 'year']
#     for field in metadata_fields:
#         if field in metadata:
#             field_value = str(metadata[field]).lower()
#             company_match = not company_name or company_name.lower() in field_value
#             year_match = not year or year in field_value
#             if company_match and year_match:
#                 return True
    
#     # Check in document content
#     content = doc.page_content.lower()
#     company_match = not company_name or company_name.lower() in content
#     year_match = not year or year in content
    
#     return company_match and year_match

# def generate(state):
#     """
#     Generate answer

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation, that contains LLM generation
#     """
#     print("---GENERATE---")
#     question = state["question"]
#     documents = state["documents"]
#     print('======STATE BEFORE GENERATION==========')
    
#     try:
#         get_object().emit("update", {
#             "username": "Generator",
#             "isAgent": False,
#             "parentAgent": "Research",
#             "content": "Generating answer based on the retrieved documents...",
#             "isUser": False,
#             "verdict": "passing to next agent",
#         })
#     except Exception as e:
#         pass
    
#     print(state)
#     print(f"Mode: {state['mode']}")
    
#     if state['mode'] == "web_search":
#         # RAG generation
#         if isinstance(documents, list):
#             context = '\n\n'.join(documents)
#         else:
#             context = documents
#         generation = rag_chain.invoke({"context": context, "question": question})
#     else:
#         if isinstance(documents, list):
#             generation = '\n\n'.join(doc for doc in documents)
#         else:
#             generation = documents
    
#     return {"documents": documents, "question": question, "generation": generation}

# def grade_documents(state):
#     """
#     Determines whether the retrieved documents are relevant to the question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """
#     print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     documents = state["documents"]

#     # Score each doc
#     filtered_docs = []
#     print('======STATE BEFORE GRADE DOCUMENTS==========')
#     for d in documents:
#         try:
#             score = retrieval_grader.invoke(
#                 {"question": question, "document": d}
#             )
#             grade = score.binary_score
#             print(f"Grade: {grade}")
#             print(f"Document preview: {d[:200]}..." if len(d) > 200 else d)
#             print("''''''''''''''''''''''''''''''''''''''")
#             if grade == "yes":
#                 print("---GRADE: DOCUMENT RELEVANT---")
#                 filtered_docs.append(d)
#             else:
#                 print("---GRADE: DOCUMENT NOT RELEVANT---")
#                 continue
#         except Exception as e:
#             print(f"Error grading document: {e}")
#             # Include document if grading fails
#             filtered_docs.append(d)
    
#     return {"documents": filtered_docs, "question": question}

# def transform_query(state):
#     """
#     Transform the query to produce a better question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates question key with a re-phrased question
#     """
#     print("---TRANSFORM QUERY---")
#     question = state["question"]
#     documents = state["documents"]

#     # Re-write question
#     better_question = question_rewriter.invoke({"question": question})
#     print(f"Better question: {better_question}")
#     print("#####################################")
#     return {"documents": documents, "question": better_question}

# def web_search(state):
#     """
#     Web search based on the re-phrased question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with appended web results
#     """
#     try:
#         get_object().emit("update", {
#             "username": "Web Search",
#             "isAgent": False,
#             "parentAgent": "Research",
#             "content": "Performing web search",
#             "isUser": False,
#             "verdict": "passing to next agent",
#         })
#     except Exception as e:
#         pass
    
#     print("---WEB SEARCH---")
#     question = state["question"]
#     state["mode"] = "web_search"
    
#     # Web search
#     docs = web_search_tool.invoke({"query": question})
#     web_results = "\n".join([d["content"] for d in docs])
#     web_results = Document(page_content=web_results)

#     return {"documents": [web_results.page_content], "question": question, "mode": "web_search"}

# #===================================QUERY REWRITER===============================================
# class RewrittenQueries(BaseModel):
#     """Possible queries for a given user question."""

#     query1: str = Field(description="Rewritten query number 1")
#     query2: str = Field(description="Rewritten query number 2")
#     query3: str = Field(description="Rewritten query number 3")
#     query4: str = Field(description="Rewritten query number 4")
#     query5: str = Field(description="Rewritten query number 5")
#     company_name: str = Field(description="Name of the company (if mentioned)")
#     year: str = Field(description="Year of the financial document (if mentioned)")
#     table: str = Field(description="Whether the answer might be in a table (YES/NO)")

# structured_llm_rewriter = llm.with_structured_output(RewrittenQueries)

# # Prompt
# system_multiple_queries = """
# You are an expert at rewriting a user question for querying a vectorstore containing financial documents.
# The database contains documents related to SEC fillings of multiple companies and other financial documents.
# Your task is to generate multiple rephrased queries for the user question to improve search results.
# While rewriting queries remember that the query text need to closely match the content of the documents in the database for vector store search.
# Output exactly 5 rephrased queries for the user question along with the company name and financial year of the document as inferred from the question.
# If you think a query might belong to a certain section of the financial document, you can include that in the query.
# If you think the answer might be in a table, set table parameter to YES, else NO.
# If no specific company or year is mentioned, return empty strings for company_name and year.
# """
# multiple_queries_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_multiple_queries),
#         ("human", "{question}"),
#     ]
# )

# query_rewriter_multi = multiple_queries_prompt | structured_llm_rewriter

# def possible_queries(state):
#     """
#     Transform the question to produce multiple rephrased queries.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates queries key with multiple rephrased queries
#     """
#     print("---REPHRASED QUERIES---")
#     question = state["question"]
#     documents = state["documents"]

#     # Re-write question
#     result = query_rewriter_multi.invoke({"question": question})
#     queries = [result.query1, result.query2, result.query3, result.query4, result.query5]
#     company_name = result.company_name
#     year = result.year
#     print(f"Rewritten queries result: {result}")
#     print("#####################################")
    
#     return {
#         "documents": documents, 
#         "queries": [question] + queries, 
#         "question": question, 
#         "company_name": company_name, 
#         "year": year, 
#         "table": result.table, 
#         "mode": "vectorstore"
#     }

# #==================================================================================
# def route_question(state):
#     """
#     Route question to web search or RAG.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Next node to call
#     """
#     print("---ROUTE QUESTION---")
#     question = state["question"]
#     state["count"] = 0
    
#     try:
#         source = question_router.invoke({"question": question})
#         if source.datasource == "web_search":
#             state['mode'] = "web_search"
#             print("---ROUTE QUESTION TO WEB SEARCH---")
#             return "web_search"
#         elif source.datasource == "vectorstore":
#             state['mode'] = "vectorstore"
#             print("---ROUTE QUESTION TO RAG---")
#             return "vectorstore"
#     except Exception as e:
#         print(f"Error in routing: {e}")
#         # Default to vectorstore
#         state['mode'] = "vectorstore"
#         return "vectorstore"

# def decide_to_generate(state):
#     """
#     Determines whether to generate an answer, or re-generate a question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Binary decision for next node to call
#     """
#     print("---ASSESS GRADED DOCUMENTS---")
#     filtered_documents = state["documents"]

#     if not filtered_documents:
#         # All documents have been filtered check_relevance
#         # We will re-generate a new query
#         print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
#         return "transform_query"
#     else:
#         # We have relevant documents, so generate answer
#         print("---DECISION: GENERATE---")
#         return "generate"

# def decide_after_transform(state):
#     print("---ASSESS TRANSFORMED QUERY DOCUMENTS---")
#     filtered_documents = state["documents"]

#     if not filtered_documents and state["count"] >= 2:
#         # All documents have been filtered, try web search
#         print("---DECISION: ALL DOCUMENTS ARE STILL NOT RELEVANT TO QUESTION, PERFORM WEB SEARCH---")
#         return "web_search"
#     else:
#         # We have relevant documents, so generate answer
#         print("---DECISION: RETRIEVE---")
#         return "retrieve"

# def grade_generation_v_documents_and_question(state):
#     """
#     Determines whether the generation is grounded in the document and answers question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Decision for next node to call
#     """
#     print("---CHECK HALLUCINATIONS---")
#     question = state["question"]
#     documents = state["documents"]
#     generation = state["generation"]

#     try:
#         score = hallucination_grader.invoke(
#             {"documents": documents, "generation": generation}
#         )
#         grade = score.binary_score

#         # Check hallucination
#         if grade == "yes":
#             print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
#             # Check question-answering
#             print("---GRADE GENERATION vs QUESTION---")
#             score = answer_grader.invoke({"question": question, "generation": generation})
#             grade = score.binary_score
#             if grade == "yes":
#                 print("---DECISION: GENERATION ADDRESSES QUESTION---")
#                 return "useful"
#             else:
#                 print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
#                 return "not useful"
#         else:
#             print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
#             return "not supported"
#     except Exception as e:
#         print(f"Error in grading generation: {e}")
#         return "useful"  # Default to useful if grading fails

# # ======================================================================================================
# workflow = StateGraph(GraphState)

# # Define the nodes
# workflow.add_node("web_search", web_search)  # web search
# workflow.add_node("retrieve", retrieve)  # retrieve
# workflow.add_node("grade_documents", grade_documents)  # grade documents
# workflow.add_node("generate", generate)  # generate
# workflow.add_node("transform_query", transform_query)  # transform_query
# workflow.add_node("possible_queries", possible_queries)  # possible_queries

# # Build graph
# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         "web_search": "web_search",
#         "vectorstore": "possible_queries",
#     },
# )

# workflow.add_edge("possible_queries", "retrieve")
# workflow.add_edge("web_search", "generate")
# workflow.add_edge("retrieve", "grade_documents")
# workflow.add_conditional_edges(
#     "grade_documents",
#     decide_to_generate,
#     {
#         "transform_query": "transform_query",
#         "generate": "generate",
#     },
# )

# workflow.add_conditional_edges(
#     "transform_query",
#     decide_after_transform,
#     {
#         "web_search": "web_search",
#         "retrieve": "retrieve",
#     },
# )

# workflow.add_edge("generate", END)

# # Compile
# app = workflow.compile()

# class DataNode(BaseModel):
#     query: str = Field(description="The Query to be processed for fetching data")

# def data_node_function(query: str) -> str:
#     """
#     An LLM agent with access to a structured tool for fetching internal data or online source.
#     """
#     try:
#         get_object().emit("update", {
#             "username": "Research",
#             "isAgent": True,
#             "parentAgent": "Supervisor",
#             "content": "Building the research agent...",
#             "isUser": False,
#             "verdict": "passing to next agent",
#         })
#     except Exception as e:
#         pass
    
#     inputs = {
#         "question": query,
#         "count": 0,
#         "documents": [],
#         "generation": "",
#         "mode": "",
#         "queries": [],
#         "company_name": "",
#         "year": "",
#         "table": ""
#     }
    
#     try:
#         results = app.invoke(inputs)
#         return results['generation']
#     except Exception as e:
#         print(f"Error in data_node_function: {e}")
#         return f"Error retrieving data: {str(e)}"

# data_node_tool = StructuredTool.from_function(
#     data_node_function,
#     name="data_node_tool",
#     description="""data_node_tool(query: str) -> str:
#     An LLM agent with access to a structured tool for fetching internal data or online source.
#     Internal data includes financial documents, SEC filings, and other financial data of various companies.
#     Use it whenever you need to fetch internal data or online source.
#     It can satisfy all your queries related to data retrieval.
#     SEARCH SPECIFIC RULES:
#         Provide concise queries to this tool, DO NOT give vague queries for search like
#         - 'What was the gdp of the US for last 5 years?'
#         - 'What is the percentage increase in Indian income in the last few years?'
#         Instead, provide specific queries like
#         - 'GDP of the US for 2020'
#         - 'Income percentage increase in India for 2019'
#         ALWAYS mention units for searching specific data wherever applicable and use uniform units for an entity across queries.
#         Eg: Always use 'USD' for currency,'percentage' for percentage, etc.
#     INTERNAL DATA SPECIFIC RULES:
#         The tool can fetch internal data like financial documents, SEC filings, and other financial data of various companies.
#         The retriever is very sensitive to the query, so if you are unable to infer from the data in 1-2 queries, keep on trying again with rephrased queries

#     ALWAYS provide specific queries to get accurate results.
#     DO NOT try to fetch multiple data points in a single query, instead, make multiple queries.
#     """,
#     args_schema=DataNode,)



# # Test the updated code
# if __name__ == "__main__":
#     print('___________________________________________')
#     query = "Which business segment negatively impacted 3M's overall growth in 2022, excluding MD&A effects?"
#     result = data_node_tool.invoke({"query": query})
#     print(result)
    
#     # Test ChromaDB search functionality
#     test_query = """
#     ### document: 3M's 2024 Annual Report
#     """
    
#     try:
#         # Test basic similarity search
#         search_results = client.similarity_search_with_score(test_query, k=5)
#         print(f"ChromaDB search results: {len(search_results)} documents found")
#         for i, (doc, score) in enumerate(search_results):
#             print(f"Document {i+1} - Score: {score}")
#             print(f"Content preview: {doc.page_content[:200]}...")
#             print("---")
#     except Exception as e:
#         print(f"ChromaDB search error: {e}")
        
        
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import WebBaseLoader  # type: ignore
from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from typing import Literal, List
from typing_extensions import TypedDict
from langchain.schema import Document  # type: ignore
from langchain import hub  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
from langgraph.graph import END, StateGraph, START  # type: ignore
from pprint import pprint
import os
from dotenv import load_dotenv  # type: ignore
from langchain.tools import StructuredTool
from langchain_community.vectorstores import Qdrant  # ✅ NEW Qdrant
from qdrant_client import QdrantClient  # ✅ Qdrant client
from qdrant_client.http import models  # ✅ Qdrant models

load_dotenv()

os.environ['OPENAI_API_KEY'] = "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"

# Initialize LLM and embeddings
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://b46a2c8f-2e88-4883-adc7-50ea89e775e3.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzUxMTQzNjgwfQ.s54bLE4-tQn0C4t-nP-tMggZl5RJsk3MlmfCjdbLfow",
)

# Collection name
COLLECTION_NAME = "pdf_documents"

# Initialize Qdrant vector store
try:
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    print(f"Connected to Qdrant collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    # Create collection if it doesn't exist
    try:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=models.Distance.COSINE
            )
        )
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
        print(f"Created and connected to Qdrant collection: {COLLECTION_NAME}")
    except Exception as create_error:
        print(f"Error creating collection: {create_error}")

# Create retriever from vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Initialize client reference for direct operations
client = vector_store

#============================= QUESTION ROUTER =================================
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to SEC fillings or financial data of multiple companies.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_router),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

#================================================================================

#============================= GRADE DOCUMENTS =================================
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    1)If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant\n
    2)It is OK if the facts have SOME information that is unrelated to the question (1) is met \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_doc_grader

#================================================================================

#==============================RAG CHAIN=========================================
# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm_mini | StrOutputParser()

#============================= GRADE HALLUCINATIONS =================================
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# LLM with function call
structured_llm_grader = llm_mini.with_structured_output(GradeHallucinations)

# Prompt
system_hallucination_grader = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_hallucination_grader),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
#==================================================================================

#============================= GRADE ANSWER =======================================
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# LLM with function call
structured_llm_answer_grader = llm_mini.with_structured_output(GradeAnswer)

# Prompt
system_answer_grader = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer_grader),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_answer_grader
#==================================================================================

#============================= QUERY REWRITER ====================================
# Prompt
system_rewriter = """
You are a question re-writer that converts an input question to a better version that is optimized \n 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
And reformulate the query to better suite the content of the documents in the vectorstore, which are mainly SEC filings and financial documents."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewriter),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

#==================================================================================

#============================= WEB SEARCH ========================================
web_search_tool = TavilySearchResults(k=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

#==================================================================================

#============================= WORKFLOW ===========================================
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        count: Number of times retriever is called
        queries: list of possible queries
    """

    question: str
    generation: str
    documents: List[str]
    count: int
    queries: List[str]
    company_name: str
    year: str
    table: str
    mode: str

from constants import get_object

def retrieve(state):
    """
    Retrieve documents using Qdrant - Simplified version without filtering issues

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    queries = state["queries"]
    count = state["count"] + 1
    print('======STATE BEFORE RETRIEVAL==========')
    
    try:
        get_object().emit("update", {
            "username": "Retriever",
            "isAgent": False,
            "parentAgent": "Research",
            "content": "Retrieving documents from Qdrant...",
            "isUser": False,
            "verdict": "passing to next agent",
        })
    except Exception as e:
        pass
    
    print(state)
    
    # Simplified retrieval using Qdrant
    documents = []
    all_results = []
    
    try:
        if queries[0] != "" and count == 1:
            # Use multiple queries
            for query in queries:
                # Search for table content
                table_query = f"Markdown Table {query}"
                try:
                    table_docs = vector_store.similarity_search_with_score(table_query, k=5)
                    for doc, score in table_docs:
                        all_results.append((doc, score, 'table'))
                except Exception as e:
                    print(f"Table search error: {e}")
                
                # Search for regular text content
                try:
                    text_docs = vector_store.similarity_search_with_score(query, k=5)
                    for doc, score in text_docs:
                        all_results.append((doc, score, 'text'))
                except Exception as e:
                    print(f"Text search error: {e}")
        else:
            # Single query search
            table_query = f"Markdown Table {question}"
            try:
                table_results = vector_store.similarity_search_with_score(table_query, k=5)
                for doc, score in table_results:
                    all_results.append((doc, score, 'table'))
            except Exception as e:
                print(f"Table search error: {e}")
            
            try:
                text_results = vector_store.similarity_search_with_score(question, k=10)
                for doc, score in text_results:
                    all_results.append((doc, score, 'text'))
            except Exception as e:
                print(f"Text search error: {e}")

        # Remove duplicates based on page_content
        unique_results = []
        seen_content = set()
        for doc, score, doc_type in all_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                unique_results.append((doc, score, doc_type))
                seen_content.add(content_hash)
        
        # Sort by similarity score (higher is better for Qdrant cosine similarity)
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        # Prioritize table results, then text results
        table_results = [r for r in unique_results if r[2] == 'table'][:3]
        text_results = [r for r in unique_results if r[2] == 'text'][:7]
        
        # Combine results
        final_results = table_results + text_results
        documents = [doc.page_content for doc, score, doc_type in final_results[:10]]
        
        print(f"Retrieved {len(documents)} documents")
        
        if len(documents) == 0:
            print("No documents found, trying basic search...")
            # Try a basic search as fallback
            try:
                basic_results = vector_store.similarity_search(question, k=10)
                documents = [doc.page_content for doc in basic_results]
                print(f"Basic search retrieved {len(documents)} documents")
            except Exception as e:
                print(f"Basic search error: {e}")
    
    except Exception as e:
        print(f"Retrieval error: {e}")
        # Fallback to basic search
        try:
            basic_results = vector_store.similarity_search(question, k=10)
            documents = [doc.page_content for doc in basic_results]
            print(f"Fallback search retrieved {len(documents)} documents")
        except Exception as fallback_error:
            print(f"Fallback search error: {fallback_error}")
            documents = []
    
    return {"documents": documents, "question": question, "count": count}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    print('======STATE BEFORE GENERATION==========')
    
    try:
        get_object().emit("update", {
            "username": "Generator",
            "isAgent": False,
            "parentAgent": "Research",
            "content": "Generating answer based on the retrieved documents...",
            "isUser": False,
            "verdict": "passing to next agent",
        })
    except Exception as e:
        pass
    
    print(state)
    print(f"Mode: {state['mode']}")
    
    if state['mode'] == "web_search":
        # RAG generation
        if isinstance(documents, list):
            context = '\n\n'.join(documents)
        else:
            context = documents
        generation = rag_chain.invoke({"context": context, "question": question})
    else:
        if isinstance(documents, list):
            generation = '\n\n'.join(doc for doc in documents)
        else:
            generation = documents
    
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    print('======STATE BEFORE GRADE DOCUMENTS==========')
    for d in documents:
        try:
            score = retrieval_grader.invoke(
                {"question": question, "document": d}
            )
            grade = score.binary_score
            print(f"Grade: {grade}")
            print(f"Document preview: {d[:200]}..." if len(d) > 200 else d)
            print("''''''''''''''''''''''''''''''''''''''")
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        except Exception as e:
            print(f"Error grading document: {e}")
            # Include document if grading fails
            filtered_docs.append(d)
    
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print(f"Better question: {better_question}")
    print("#####################################")
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    try:
        get_object().emit("update", {
            "username": "Web Search",
            "isAgent": False,
            "parentAgent": "Research",
            "content": "Performing web search",
            "isUser": False,
            "verdict": "passing to next agent",
        })
    except Exception as e:
        pass
    
    print("---WEB SEARCH---")
    question = state["question"]
    state["mode"] = "web_search"
    
    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": [web_results.page_content], "question": question, "mode": "web_search"}

#===================================QUERY REWRITER===============================================
class RewrittenQueries(BaseModel):
    """Possible queries for a given user question."""

    query1: str = Field(description="Rewritten query number 1")
    query2: str = Field(description="Rewritten query number 2")
    query3: str = Field(description="Rewritten query number 3")
    query4: str = Field(description="Rewritten query number 4")
    query5: str = Field(description="Rewritten query number 5")
    company_name: str = Field(description="Name of the company (if mentioned)")
    year: str = Field(description="Year of the financial document (if mentioned)")
    table: str = Field(description="Whether the answer might be in a table (YES/NO)")

structured_llm_rewriter = llm.with_structured_output(RewrittenQueries)

# Prompt
system_multiple_queries = """
You are an expert at rewriting a user question for querying a vectorstore containing financial documents.
The database contains documents related to SEC fillings of multiple companies and other financial documents.
Your task is to generate multiple rephrased queries for the user question to improve search results.
While rewriting queries remember that the query text need to closely match the content of the documents in the database for vector store search.
Output exactly 5 rephrased queries for the user question along with the company name and financial year of the document as inferred from the question.
If you think a query might belong to a certain section of the financial document, you can include that in the query.
If you think the answer might be in a table, set table parameter to YES, else NO.
If no specific company or year is mentioned, return empty strings for company_name and year.
"""
multiple_queries_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_multiple_queries),
        ("human", "{question}"),
    ]
)

query_rewriter_multi = multiple_queries_prompt | structured_llm_rewriter

def possible_queries(state):
    """
    Transform the question to produce multiple rephrased queries.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates queries key with multiple rephrased queries
    """
    print("---REPHRASED QUERIES---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    result = query_rewriter_multi.invoke({"question": question})
    queries = [result.query1, result.query2, result.query3, result.query4, result.query5]
    company_name = result.company_name
    year = result.year
    print(f"Rewritten queries result: {result}")
    print("#####################################")
    
    return {
        "documents": documents, 
        "queries": [question] + queries, 
        "question": question, 
        "company_name": company_name, 
        "year": year, 
        "table": result.table, 
        "mode": "vectorstore"
    }

#==================================================================================
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    state["count"] = 0
    
    try:
        source = question_router.invoke({"question": question})
        if source.datasource == "web_search":
            state['mode'] = "web_search"
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            state['mode'] = "vectorstore"
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    except Exception as e:
        print(f"Error in routing: {e}")
        # Default to vectorstore
        state['mode'] = "vectorstore"
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def decide_after_transform(state):
    print("---ASSESS TRANSFORMED QUERY DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents and state["count"] >= 2:
        # All documents have been filtered, try web search
        print("---DECISION: ALL DOCUMENTS ARE STILL NOT RELEVANT TO QUESTION, PERFORM WEB SEARCH---")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RETRIEVE---")
        return "retrieve"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    try:
        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    except Exception as e:
        print(f"Error in grading generation: {e}")
        return "useful"  # Default to useful if grading fails

# ======================================================================================================
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("possible_queries", possible_queries)  # possible_queries

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "possible_queries",
    },
)

workflow.add_edge("possible_queries", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "transform_query",
    decide_after_transform,
    {
        "web_search": "web_search",
        "retrieve": "retrieve",
    },
)

workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

class DataNode(BaseModel):
    query: str = Field(description="The Query to be processed for fetching data")

def data_node_function(query: str) -> str:
    """
    An LLM agent with access to a structured tool for fetching internal data or online source.
    """
    try:
        get_object().emit("update", {
            "username": "Research",
            "isAgent": True,
            "parentAgent": "Supervisor",
            "content": "Building the research agent...",
            "isUser": False,
            "verdict": "passing to next agent",
        })
    except Exception as e:
        pass
    
    inputs = {
        "question": query,
        "count": 0,
        "documents": [],
        "generation": "",
        "mode": "",
        "queries": [],
        "company_name": "",
        "year": "",
        "table": ""
    }
    
    try:
        results = app.invoke(inputs)
        return results['generation']
    except Exception as e:
        print(f"Error in data_node_function: {e}")
        return f"Error retrieving data: {str(e)}"

data_node_tool = StructuredTool.from_function(
    data_node_function,
    name="data_node_tool",
    description="""data_node_tool(query: str) -> str:
    An LLM agent with access to a structured tool for fetching internal data or online source.
    Internal data includes financial documents, SEC filings, and other financial data of various companies.
    Use it whenever you need to fetch internal data or online source.
    It can satisfy all your queries related to data retrieval.
    SEARCH SPECIFIC RULES:
        Provide concise queries to this tool, DO NOT give vague queries for search like
        - 'What was the gdp of the US for last 5 years?'
        - 'What is the percentage increase in Indian income in the last few years?'
        Instead, provide specific queries like
        - 'GDP of the US for 2020'
        - 'Income percentage increase in India for 2019'
        ALWAYS mention units for searching specific data wherever applicable and use uniform units for an entity across queries.
        Eg: Always use 'USD' for currency,'percentage' for percentage, etc.
    INTERNAL DATA SPECIFIC RULES:
        The tool can fetch internal data like financial documents, SEC filings, and other financial data of various companies.
        The retriever is very sensitive to the query, so if you are unable to infer from the data in 1-2 queries, keep on trying again with rephrased queries

    ALWAYS provide specific queries to get accurate results.
    DO NOT try to fetch multiple data points in a single query, instead, make multiple queries.
    """,
    args_schema=DataNode,)

# Test the updated code
if __name__ == "__main__":
    print('___________________________________________')
    query = "what is 3M???
    "
    result = data_node_tool.invoke({"query": query})
    print(result)
    
    # Test Qdrant search functionality
    test_query = "3M business segments 2022"
    
    try:
        # Test basic similarity search
        search_results = vector_store.similarity_search_with_score(test_query, k=5)
        print(f"Qdrant search results: {len(search_results)} documents found")
        for i, (doc, score) in enumerate(search_results):
            print(f"Document {i+1} - Score: {score}")
            print(f"Content preview: {doc.page_content[:200]}...")
            print("---")
    except Exception as e:
        print(f"Qdrant search error: {e}")
