import requests
import os
from decouple import config
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import BaseTool
from typing import List, Any
from time import time

os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")

class GetModel1Tool(BaseTool):
    name: str = "GetModel1Tool"
    description: str =  "Use this to get the answer from model 1"

    def _run(self, input:int) -> Any:
        response = requests.get(f"http://127.0.0.1:8000/api/model1/{input}")
        response.raise_for_status()  # Raise an error for HTTP errors
        return response.json()
    
    def _arun(self, input:int) -> Any:
        raise NotImplementedError("This tools does not suport async")
    
class GetModel2Tool(BaseTool):
    name: str  = "GetModel2Tool"
    description: str  = "Use this to get the answer from model 2"

    def _run(self, input:int) -> Any:
        response = requests.get(f"http://127.0.0.1:8000/api/model2/{input}")
        response.raise_for_status()  # Raise an error for HTTP errors
        return response.json()
    
    def _arun(self, input:int) -> Any:
        raise NotImplementedError("This tools does not suport async")
    
get_model_1 = GetModel1Tool()
get_model_2 = GetModel2Tool()

def get_routine():
    '''
    Chose a routine and use it
    '''
    routine = """
    Use the input 5 (integer) in the first model, and then utilize the result to input to second model.
    What is the result?

    Make sure the input to both models are integers.
    """

    entry = 12

    routine = f"""
    Make the routine described above using model 1 and model 2

    Entry -> Model 1 -> Model 2 -> Model 2 -> Model 1 -> Output

    The entry for this problem is {entry} (integer)
    Give me the output as integer
    """

    return routine

def get_llm() -> ChatGroq:
    print("Getting model...")
    
    return ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )

def get_tool() -> list:
    """Method to make it easier to change the search tool"""
    print("Getting tools...")
    
    return [get_model_1, get_model_2]

def get_agent_executor(llm, tools: List[Any]):
    """The react agent to unify the model and the tools"""
    print("Creating agent executor...")
    react_instructions = hub.pull('hwchase17/react')

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_instructions,
    )
            
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=None
    )

agent_executor = get_agent_executor(get_llm(), tools=get_tool())
routine = get_routine()

os.system("clear")
print(f"Question: {routine}")

s = time()

output = agent_executor.invoke({"input": routine},)

print(output.get("output"))
e = time()
print(f'Answered in {e-s:.2f} seconds')