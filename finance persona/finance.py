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

# function to use persona data for each diff category
def get_personas_by_category(data,category):
    if category in data:
        supervisor=data[category]["supervisor_persona"]
        agent_1=data[category]["agent_1_persona"]
        agent_2=data[category]["agent_2_persona"]
        # Return the personas as a dictionary
        return {
            "supervisor": supervisor,
            "agent_1": agent_1,
            "agent_2": agent_2
        }
    else:
        raise ValueError(f"Category '{category}' not found in the data.")

json_file_path = "finance_persona.json"
# Load the persona data from the JSON file
persona_data = load_persona_json(json_file_path)


#-----------------------------------MAIN FUNCTIONALITY-------------------------------------------------------------

llm_4o_mini= ChatOpenAI(model="gpt-4o-mini",)

llm= ChatOpenAI(model="gpt-4o")

class SupervisorReply(BaseModel):
    """
    response from the supervisor
    """
    content:str= Field(description="The content of the response")
    fallback: str = Field(description="A fallback flag when the conversation ends. Can be only YES or NO")

class AgentReply(BaseModel):
    """
    response from the agent
    """
    content: str = Field(description="The content of the response")
    return_summary: str = Field(description="A flag to end the conversation and return a summary. Can be only YES or NO")


structured_llm_mini=llm_4o_mini.with_structured_output(AgentReply)
structured_llm=llm.with_structured_output(SupervisorReply)


# state definition to keep track of the conversation
class State(TypedDict):
    """a state object to keep track of the conversation"""
    topic: str= Field(description="The topic of the conversation with additional context.")
    context:str= Field(description="Additional context or data that needs to be passed to the agents for conversation.")
    conversation:Annotated[list,add_messages]# adding meta data to the conversation
    supervisor_:str=Field(description="the persona of supervisor agent.")
    agent1:str=Field(description="the persona of agent 1.")
    agent2:str=Field(description="the persona of agent 2.")
    summary:str=Field(description="A summary of the conversation.")
    fallback:str=Field(description="A fallback message when the conversation ends. Can be only YES or NO")
    return_summary: str=Field(description="A return summary message when the summary is generated. Can be only YES or NO")
    count:int=Field(description="The count of the conversation messages.")
    
def agent_node_1(state:State)->State:
    """ a node that represents the first agent in the conversation"""
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", state["agent1"] + "You can set the fallback to YES or NO based on wether you need to return the conversation to the supervisor or not. YES to return to supervisor, NO to continue the conversation.  Set the fallback to YES if the conversation is becoming too long and you need the supervisor to summarize the conversation."),
            MessagesPlaceholder(variable_name="messages"),

        ]
    )
    agent_1=prompt | structured_llm_mini
    if state["count"]==0:
        messages=[
            HumanMessage(f"Your task is to simulate an agent in a financial conversation. You need to plan your part in a conversation on the topic:\n {state['topic']} and additional context:\n{state['context']}. \n Initiate the conversation as the given persona.")
        ]
        response=agent_1.invoke({"messages": messages})
        state["count"]+=1
        state["fallback"]=response.fallback
        state["conversation"].append(response.content)
        print(f"Agent 1: {response.fallback}")
        
        
    else:
        state["conversation"][-1]=HumanMessage(content=state["conversation"][-1].content)
        response=agent_1.invoke({"messages": state["conversation"]})
        state["count"]+=1
        state["fallback"]=response.fallback
        state["conversation"].append(response.content)
        print(f"Agent 1: {response.fallback}")
    return state

def agent_node_2(state:State)->State:
    """a node that represents the second agent in the conversation"""
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", state["agent2"]  + "You can set the fallback to YES or NO based on wether you need to return the conversation to the supervisor or not. YES to return to supervisor, NO to continue the conversation. Set the fallback to YES if the conversation is becoming too long and you need the supervisor to summarize the conversation."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    messages=state['conversation']
    agent_2=prompt | structured_llm_mini
    print(state)
    response=agent_2.invoke({"messages": state["conversation"]})
    state["count"]+=1
    state["fallback"]=response.fallback
    state["conversation"].append(response.content)
    print(f"Agent 2: {response.fallback}")
    return state

