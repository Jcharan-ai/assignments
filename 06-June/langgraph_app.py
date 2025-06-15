import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import operator
from typing import TypedDict,Optional, List
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

# --- 1. Define the Graph State ---
# This defines the schema of the state that will be passed between nodes.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's initial question.
        context: Accumulated context/information from various nodes.
        generation: The LLM's generated output.
        validation_result: Result of the validation (True/False).
        tool_calls: List of tools the LLM has decided to call.
        messages: List of messages in the conversation (for conversational agents).
        next_node: The next node to execute, determined by the supervisor.
    """
    question: str
    context: Optional[str] = None
    generation: Optional[str] = None
    validation_result: Optional[bool] = None
    tool_calls: Optional[List[dict]] = None
    messages: List[BaseMessage]
    next_node: Optional[str] = None # Used by supervisor to direct flow

# --- 2. Define the LLM ---
model=ChatGroq(model_name="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))


# --- 3. Define the RAG Context and format docs ---
def get_context(state: GraphState) -> str:
    """
    Retrieves the context from the state.
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    loader=DirectoryLoader("./data",glob="./*.txt",loader_cls=TextLoader)
    docs=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    new_docs=text_splitter.split_documents(documents=docs)
    db=FAISS.from_documents(new_docs, embeddings)
    retriever=db.as_retriever(search_kwargs={"k": 2})
    respose = retriever.invoke(state['question'])
    
    state.context = respose.page_content
    state['messages'].append(AIMessage(content=f"RAG result: {respose}"))
    # If the context is None, return an empty string
    return state.context if state.context else ""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 4. Define Nodes ---

# 4.1. Supervisor Node
# This node uses an LLM to decide the next action based on the current state.
# It's like a central router.
def supervisor_node(state: GraphState) -> GraphState:
    print("--- SUPERVISOR NODE ---")
    messages = state['messages']
    question = state['question']

    # If this is the first turn, or a validation failed, we need to decide.
    if state.get("validation_result") is False or len(messages) == 1: # Initial call or validation failed
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful assistant. Based on the user's question and current context,
             decide the next best action.
             Possible actions are:
             - 'llm_call': If you need to generate a response directly.
             - 'rag_node': If the question requires information from an internal knowledge base.
             - 'web_crawler_node': If the question requires real-time information from the internet.
             - 'END': If the question is fully answered and validated.

             Current question: {question}
             Current context: {context}
             Previous generation: {generation}
             Validation result: {validation_result}

             Instruct the system to call a specific node by returning ONLY the node name (e.g., 'llm_call', 'rag_node', 'web_crawler_node', 'END').
             If a tool is needed, also suggest the tool name in your reasoning.
             """
            ),
            HumanMessage(content=f"User's query: {question}")
        ])
        chain = prompt | model
        response = chain.invoke({"question": question, "context": state.get("context", ""), "generation": state.get("generation", ""), "validation_result": state.get("validation_result")})
        llm_decision = response.content.strip().lower()
        if "rag_node" in llm_decision:
            state['next_node'] = 'rag_node'
            print("Supervisor decided: rag_node (based on general LLM content)")
        elif "web_crawler_node" in llm_decision:
            state['next_node'] = 'web_crawler_node'
            print("Supervisor decided: web_crawler_node (based on general LLM content)")
        elif "llm_call" in llm_decision:
            state['next_node'] = 'llm_call'
            print("Supervisor decided: llm_call (based on general LLM content)")
        elif "end" in llm_decision:
            state['next_node'] = 'END'
            print("Supervisor decided: END")
        else:
            # Default to LLM call if no specific directive
            state['next_node'] = 'llm_call'
            print("Supervisor defaulted to: llm_call")
    else:
        # If validation passed, we are done
        state['next_node'] = 'END'
        print("Supervisor decided: END (validation passed)")        
    return state
            
# 4.2. LLM Call Node
def llm_node(state: GraphState) -> GraphState:
    print("--- LLM NODE ---")
    messages = state['messages']
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Generate a concise and accurate response."),
        HumanMessage(content=state['question']) # Use the initial question for the generation
    ])
    chain = prompt | model
    response = chain.invoke({"question": state['question']})
    state['generation'] = response.content
    state['messages'].append(AIMessage(content=response.content))
    print(f"LLM Generated: {response.content}")
    return state

# 4.3. RAG Node
def rag_node(state: GraphState) -> GraphState:
    print("--- RAG NODE ---")
    
    # Find the rag_tool call, if any, and extract arguments
    retriever = get_context(state) # Get context from the state
     # After getting RAG data, we often want the LLM to synthesize it.
    # So, the RAG node will typically transition to the LLM node.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Based on the following RAG context, answer the user's question:\n\n{context}"),
        HumanMessage(content=state['question'])
    ])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    response = rag_chain.invoke({"context": state['context'], "question": state['question']})
    state['generation'] = response.content
    state['messages'].append(AIMessage(content=response.content))
    print(f"LLM Generated (after RAG): {response.content}")

    return state

# 4.4. Web Crawler Node
def web_crawler_node(state: GraphState) -> GraphState:
    print("--- WEB CRAWLER NODE ---")
    pass

# 4.5. Validation Node
def validation_node(state: GraphState) -> GraphState:
    print("--- VALIDATION NODE ---")
    generated_output = state.get('generation')
    question = state['question']

    if not generated_output:
        state['validation_result'] = False
        print("Validation Failed: No output generated.")
        return state
    
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"""You are a strict validator. Your task is to determine if the generated output
         satisfactorily answers the user's original question based on the provided context.
         
         Original Question: '{question}'
         Generated Output: '{generated_output}'
         Context Used: '{state.get('context', 'None')}'

         Respond with 'TRUE' if the output is valid, accurate, complete, and directly answers the question.
         Respond with 'FALSE' if the output is invalid, incomplete, inaccurate, or doesn't directly answer the question.
         Do not add any other text.
         """
        )
    ])
    validation_chain = validation_prompt | model
    validation_response = validation_chain.invoke({})
    
    validation_decision = validation_response.content.strip().upper()
    if "TRUE" in validation_decision:
        state['validation_result'] = True
        print("Validation Result: TRUE")
    else:
        state['validation_result'] = False
        print("Validation Result: FALSE (LLM based)")

    return state
# --- 5. Build the LangGraph ---

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("llm_call", llm_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("web_crawler_node", web_crawler_node)
workflow.add_node("validation_node", validation_node)

# Set the entry point
workflow.set_entry_point("supervisor")

# Define edges (transitions)

# Supervisor's role is to direct to the initial tool/LLM call
# It uses 'next_node' in the state to determine the next step.
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state['next_node'], # This function tells LangGraph which edge to follow
    {
        "llm_call": "llm_call",
        "rag_node": "rag_node",
        "web_crawler_node": "web_crawler_node",
        "end_flow": END # If no next_node is set, we end the flow
    }
)

# After LLM, RAG, or Web Crawler, always go to validation
workflow.add_edge("llm_call", "validation_node")
workflow.add_edge("rag_node", "validation_node")
workflow.add_edge("web_crawler_node", "validation_node")
workflow.add_edge("validation_node",END)

# From validation, decide whether to go back to supervisor or end
def route_validation(state: GraphState) -> str:
    if state['validation_result']:
        print("Validation PASSED. Ending process.")
        return "end_flow" # End the graph execution
    else:
        print("Validation FAILED. Returning to supervisor for re-evaluation.")
        state['messages'].append(AIMessage(content="Validation failed. Re-evaluating..."))
        return "supervisor" # Go back to supervisor
workflow.add_conditional_edges(
    "validation_node",
    route_validation,
    {
        "supervisor": "supervisor",
        "end_flow": END
    }
)

        
# Compile the graph
app=workflow.compile()

# --- 6. Example Usage ---

def run_pipeline(question: str):
    print(f"\n--- Running Pipeline for Question: '{question}' ---")
    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)]
    }
    
    # We will iterate through the graph execution until it reaches an "end" state
    final_output = None
    for s in app.stream(initial_state):
        if "__end__" in s:
            final_output = s["__end__"]
            break # Exit loop when the graph indicates completion
        # print(s) # Optional: print intermediate states for debugging

    if final_output:
        print("\n--- FINAL GENERATED OUTPUT ---")
        print(final_output.get("generation"))
        print(f"Final Validation Result: {final_output.get('validation_result')}")
    else:
        print("\n--- Pipeline did not reach a final output state. ---")

# --- Test Cases ---

# Test Case 1: Simple LLM call, expected to pass validation
run_pipeline("What is the llm model name used?")

# Modify rag_tool or validation_node to force a failure.
run_pipeline("Tell me about the Battle of Gettysburg, but make sure to mention specific dates and key figures.")

# Test Case 3: Web Crawler needed (simulated for current events)
# This might fail validation if the web data is too generic.
run_pipeline("What is the current weather in Paris?")