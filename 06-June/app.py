from langgraph import Graph, StateDict, START
from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import tool
from langchain_core.runnables import RunnableLambda
import os

# Set your OpenAI API key (replace with your actual key or environment variable)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

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


# --- 2. Initialize LLM (for LLM Node and Supervisor) ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# --- 3. Define Tool Functions (for Web Crawler and RAG - conceptual) ---

# 3.1. Web Crawler (Conceptual Tool)
# In a real scenario, this would use libraries like BeautifulSoup, Playwright, or requests.
@tool
def web_crawler_tool(query: str) -> str:
    """
    Fetches real-time information from the internet based on a query.
    Simulated by returning a hardcoded string for demonstration.
    """
    print(f"--- Calling Web Crawler for: {query} ---")
    # In a real application, you'd use a library like 'requests' and 'BeautifulSoup'
    # or a search API (e.g., Google Search API, Brave Search API) here.
    return f"Information about '{query}' from a simulated web search: [Real-time data on {query}]"

# 3.2. RAG (Conceptual Tool)
# In a real scenario, this would involve a vector database (e.g., Chroma, Pinecone)
# and an embedding model.
@tool
def rag_tool(query: str) -> str:
    """
    Retrieves information from an internal knowledge base using RAG.
    Simulated by returning a hardcoded string for demonstration.
    """
    print(f"--- Calling RAG for: {query} ---")
    # In a real application, you'd perform a vector similarity search
    # against your indexed documents.
    return f"Information about '{query}' from internal knowledge base: [Relevant document content for {query}]"

# List of tools available to the LLM
tools = [web_crawler_tool, rag_tool]

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
        chain = prompt | llm.bind_tools(tools)
        response = chain.invoke({"question": question, "context": state.get("context", ""), "generation": state.get("generation", ""), "validation_result": state.get("validation_result")})
        tool_calls = response.tool_calls

        # If LLM suggests a tool, use that for routing
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'web_crawler_tool':
                    state['next_node'] = 'web_crawler_node'
                    state['tool_calls'] = [tc] # Pass the tool call for the next node
                    print(f"Supervisor decided: web_crawler_node with tool call: {tc}")
                    return state
                elif tc['name'] == 'rag_tool':
                    state['next_node'] = 'rag_node'
                    state['tool_calls'] = [tc] # Pass the tool call for the next node
                    print(f"Supervisor decided: rag_node with tool call: {tc}")
                    return state
        
        # If no specific tool, try to determine based on LLM's general response
        # This is a fallback; usually, tools are preferred for specific actions.
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
    chain = prompt | llm
    response = chain.invoke({"question": state['question']})
    state['generation'] = response.content
    state['messages'].append(AIMessage(content=response.content))
    print(f"LLM Generated: {response.content}")
    return state

# 4.3. RAG Node
def rag_node(state: GraphState) -> GraphState:
    print("--- RAG NODE ---")
    question = state['question']
    tool_calls = state.get('tool_calls', [])

    # Find the rag_tool call, if any, and extract arguments
    rag_query = question # Default to question if no specific tool call arg
    for tc in tool_calls:
        if tc['name'] == 'rag_tool' and 'query' in tc['args']:
            rag_query = tc['args']['query']
            break

    rag_output = rag_tool.invoke({"query": rag_query}) # Invoke the tool
    state['context'] = rag_output
    state['messages'].append(AIMessage(content=f"RAG result: {rag_output}"))
    print(f"RAG Context: {rag_output}")

    # After getting RAG data, we often want the LLM to synthesize it.
    # So, the RAG node will typically transition to the LLM node.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Based on the following RAG context, answer the user's question:\n\n{context}"),
        HumanMessage(content=state['question'])
    ])
    chain = prompt | llm
    response = chain.invoke({"context": rag_output, "question": state['question']})
    state['generation'] = response.content
    state['messages'].append(AIMessage(content=response.content))
    print(f"LLM Generated (after RAG): {response.content}")

    return state

# 4.4. Web Crawler Node
def web_crawler_node(state: GraphState) -> GraphState:
    print("--- WEB CRAWLER NODE ---")
    question = state['question']
    tool_calls = state.get('tool_calls', [])

    # Find the web_crawler_tool call, if any, and extract arguments
    web_query = question # Default to question if no specific tool call arg
    for tc in tool_calls:
        if tc['name'] == 'web_crawler_tool' and 'query' in tc['args']:
            web_query = tc['args']['query']
            break

    web_output = web_crawler_tool.invoke({"query": web_query}) # Invoke the tool
    state['context'] = web_output
    state['messages'].append(AIMessage(content=f"Web Crawler result: {web_output}"))
    print(f"Web Crawler Context: {web_output}")

    # After getting web data, we often want the LLM to synthesize it.
    # So, the Web Crawler node will typically transition to the LLM node.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Based on the following web search results, answer the user's question:\n\n{context}"),
        HumanMessage(content=state['question'])
    ])
    chain = prompt | llm
    response = chain.invoke({"context": web_output, "question": state['question']})
    state['generation'] = response.content
    state['messages'].append(AIMessage(content=response.content))
    print(f"LLM Generated (after Web Crawler): {response.content}")
    return state

# 4.5. Validation Node
def validation_node(state: GraphState) -> GraphState:
    print("--- VALIDATION NODE ---")
    generated_output = state.get('generation')
    question = state['question']

    if not generated_output:
        state['validation_result'] = False
        print("Validation Failed: No output generated.")
        return state

    # --- Explore Validation Part ---
    # This is the crucial part where you define your validation logic.
    # Examples of validation:

    # 1. Keyword/Pattern Matching:
    #    Does the output contain certain keywords?
    #    `if "sorry" in generated_output.lower(): state['validation_result'] = False`

    # 2. Length Check:
    #    Is the output of a reasonable length?
    #    `if len(generated_output) < 20: state['validation_result'] = False`

    # 3. LLM-based Validation (Recommended for robustness):
    #    Use another LLM call to critically assess the generated output.
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
    validation_chain = validation_prompt | llm
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

workflow = Graph(GraphState)

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
        "END": START # If supervisor decides END at the start, this handles it.
                     # In our flow, supervisor decides END after validation success.
    }
)

# After LLM, RAG, or Web Crawler, always go to validation
workflow.add_edge("llm_call", "validation_node")
workflow.add_edge("rag_node", "validation_node")
workflow.add_edge("web_crawler_node", "validation_node")

# From validation, decide whether to go back to supervisor or end
def route_validation(state: GraphState) -> str:
    if state['validation_result']:
        print("Validation PASSED. Ending process.")
        return "end" # End the graph execution
    else:
        print("Validation FAILED. Returning to supervisor for re-evaluation.")
        state['messages'].append(AIMessage(content="Validation failed. Re-evaluating..."))
        return "supervisor" # Go back to supervisor
workflow.add_conditional_edges(
    "validation_node",
    route_validation,
    {
        "supervisor": "supervisor",
        "end": START # Represents the final output (implicitly handled by LangGraph exiting the loop)
    }
)

# Compile the graph
app = workflow.compile()

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
run_pipeline("What is the capital of France?")

# Test Case 2: RAG needed (simulated failure or specific content)
# This might fail validation if the RAG data doesn't fully answer.
# Modify rag_tool or validation_node to force a failure.
run_pipeline("Tell me about the Battle of Gettysburg, but make sure to mention specific dates and key figures.")

# Test Case 3: Web Crawler needed (simulated for current events)
# This might fail validation if the web data is too generic.
run_pipeline("What is the current weather in Paris?")

# Test Case 4: Designed to fail validation initially and loop back
# Let's make the LLM call generate something too short, then validation should fail.
# For this, we'll temporarily adjust `llm_node` or `validation_node` for specific behavior.
# For demo purposes, I'll simulate a failure.
# To make this consistently fail, you might need to adjust the prompt for llm_node
# to intentionally generate short output, or modify the LLM-based validation prompt
# to be stricter for this specific question.

# Let's simulate a failed validation by having the supervisor initially try LLM,
# then LLM's output is short, fails validation, and supervisor tries RAG.
# This requires a more complex supervisor state management or a very strict validator.
# For now, let's just make the validation stricter if possible.

# Example of how validation might fail and loop:
# If you run the "What is the capital of France?" again, but make the validation node
# always return False for demonstration purposes:
# def validation_node(state: GraphState) -> GraphState:
#     print("--- VALIDATION NODE (FORCED FAILURE) ---")
#     state['validation_result'] = False
#     print("Validation Result: FALSE (FORCED)")
#     return state
# (Don't uncomment the above, it's just for illustration).

# Let's refine the validation for a specific test case to demonstrate the loop.
# We'll make the LLM generate a short answer for "Explain quantum entanglement briefly"
# and the validator will check for a minimum length.

def run_pipeline_with_forced_validation(question: str):
    print(f"\n--- Running Pipeline for Question: '{question}' (with potential forced validation failure) ---")
    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)]
    }
    
    # We will iterate through the graph execution until it reaches an "end" state
    final_output = None
    for s in app.stream(initial_state):
        if "__end__" in s:
            final_output = s["__end__"]
            break
        # print(s) # Optional: print intermediate states for debugging
        # You can add logic here to inspect `s` and see the current node

    if final_output:
        print("\n--- FINAL GENERATED OUTPUT ---")
        print(final_output.get("generation"))
        print(f"Final Validation Result: {final_output.get('validation_result')}")
    else:
        print("\n--- Pipeline did not reach a final output state. ---")

# To reliably demonstrate the loop, you might need to make the `validation_node`
# more specific to the content of the `generation`.
# For instance, if `generation` is very short, fail it.

# Let's add a small modification to `validation_node` for this example to illustrate.
# (This is just for demonstrating the loop back. In production, rely on robust LLM-based validation)
_original_validation_node = validation_node
def modified_validation_node(state: GraphState) -> GraphState:
    if "briefly" in state['question'].lower() and len(state.get('generation', '')) < 50:
        print("--- VALIDATION NODE (TRIGGERED SHORT OUTPUT FAILURE) ---")
        state['validation_result'] = False
        print("Validation Result: FALSE (Output too short for 'briefly' question)")
        return state
    return _original_validation_node(state) # Fallback to original validation

#Temporarily override the node in the graph for this specific demonstration.
# In a real setup, you'd manage this with different graph definitions or more complex node logic.
# This direct replacement won't work cleanly for already compiled graphs.
# Instead, for a concrete example of re-evaluation, we'll trust the LLM-based validation.

# Example for re-evaluation scenario:
# The `validation_node`'s LLM-based check is robust. If the LLM generates a poor
# answer for "What is the capital of France, but make sure to mention specific demographics?",
# the validator should catch it and send it back to the supervisor.
# The supervisor then needs to decide what to do next.

# Let's create a more challenging question for the LLM to illustrate re-evaluation.
# This assumes the LLM might initially provide a generic answer, and validation fails.
run_pipeline("Explain the concept of 'black holes' in simple terms, but make sure to include information about their formation and detection methods.")