from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langsmith import trace
from pydantic import BaseModel
from src.agent.setup_tools import tools
from typing_extensions import (
    Annotated,
    Sequence,
    TypedDict,
)
import json
from dotenv import load_dotenv
import os
import time

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# memory = MemorySaver()

class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentOutput(BaseModel):
    text_output: str
    graph_code: str  # Accept both dict and JSON string.


model = ChatOpenAI(model="gpt-4.1", temperature=0,openai_api_key=api_key)

tools_by_name = {}

model = model.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

def _latest_tool_payload(messages):
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.content and '"sources"' in m.content:
            try:
                return json.loads(m.content)
            except Exception:
                return None
    return None

def render_sources_block(messages) -> str:
    data = _latest_tool_payload(messages)
    if not data or not data.get("sources"):
        return ""
    lines = ["\nSources:"]
    for s in data["sources"]:
        i = s.get("i")
        title = s.get("title") or "Source"
        url = s.get("url") or ""
        lines.append(f"- [{i}] {title} — {url}")
    return "\n".join(lines)


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 25:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-20]]}

# Define our tool node
async def tool_node(state: AgentState):
    outputs = []
    with trace("tools_exec"):
        t0 = time.perf_counter()
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        t1 = time.perf_counter()
        print(f"Tool call took {t1 - t0:.2f} seconds")
        # trace.log({"tool_call_ms": round((t1-t0)*1000)})
    return {"messages": outputs}

# Define the node that calls the model
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # this is similar to customizing the create_react_agent with state_modifier, but is a lot more flexible
    system_prompt = SystemMessage(content=(
    "You are a helpful assistant with access to tools.\n"
    "When the user asks about internal docs/policies/handbooks, call the tool "
    "`generate_retrieval_response` first.\n"

    """   When you call the `generate_retrieval_response` tool:

    • Always include: user_groups=["all"], sources=["gitlab-handbook"].
    • Consider adding at most TWO section groups (sp1_any) **iff** the query clearly names a domain.
    • If unsure, omit sp1_any (unfiltered search). Do NOT invent values.

    Allowed section groups (sp1_any): {SECTION_PREFIX1_MENU}
    Return ONLY JSON with these keys:
    {"user_query": "...", "k": 15, "top": 6, "max_per_doc": 2,
    "user_groups": ["all"], "sources": ["gitlab-handbook"],
    "sp1_any": [], "sp2_any": [], "tags_all": [], "tags_any": [],
    "updated_after": null, "updated_before": null}
    """

    """
    User: "What is GitLab's parental leave policy?"
    Tool args JSON:
    {"user_query":"What is GitLab's parental leave policy?",
    "user_groups":["all"], "sources":["gitlab-handbook"],
    "sp1_any":["people-group","total-rewards"]}

    User: "SEV1 vs SEV2 definitions"
    Tool args JSON:
    {"user_query":"SEV1 vs SEV2 definitions",
    "user_groups":["all"], "sources":["gitlab-handbook"],
    "sp1_any":["security"]}

    User: "How do I expense a laptop?"
    Tool args JSON:
    {"user_query":"How do I expense a laptop?",
    "user_groups":["all"], "sources":["gitlab-handbook"],
    "sp1_any":["finance","business-technology"]}

    User: "OKR review cadence"
    Tool args JSON:
    {"user_query":"OKR review cadence",
    "user_groups":["all"], "sources":["gitlab-handbook"],
    "sp1_any":["product-development"]}

    User: "help"
    Tool args JSON:
    {"user_query":"help",
    "user_groups":["all"], "sources":["gitlab-handbook"]}
    """

    "The tool returns JSON with keys:\n"
    "  - 'stitched': ranked context blocks\n"
    "  - 'sources': list of {i, title, url, doc_id, span, score}\n"
    "Answer strictly from 'stitched'. Use inline citations like [n] matching 'sources[i-1]'.\n"
    "At the end of your answer, add a 'Sources' section rendered as bullets where each bullet is:\n"
    "  [n] {title} — {url}\n"
    "Use the exact 'url' field from 'sources' so links are clickable.\n"
    "If the tool returns no results, say you don’t have enough information and suggest one follow-up question.\n"
    "Keep answers concise and accurate."
))
    with trace("agent_llm"):
        t0 = time.perf_counter()
        response = model.invoke([system_prompt] + state["messages"], config)
        t1 = time.perf_counter()
        print(f"LLM call took {t1 - t0:.2f} seconds")
        # trace.log({"agent_llm_ms": round((t1-t0)*1000)})
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "delete_messages" # "end"
    # Otherwise if there is, we continue
    else:
        return "tools" # "continue"

# Define a new graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# This is our new node we're defining
# workflow.add_node(delete_messages)
workflow.add_node("delete_messages", delete_messages)

# Set the entrypoint as `agent`
# This means that this node is the first one called
# workflow.set_entry_point("agent")
workflow.add_edge(START, "agent")


# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        "tools": "tools",
        "delete_messages": "delete_messages"
    }
    # {
    #     # If `tools`, then we call the tool node.
    #     "continue": "tools",
    #     # Otherwise we finish.
    #     "end": END,
    # },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# This is the new edge we're adding: after we delete messages, we finish
workflow.add_edge("delete_messages", END)
# Now we can compile and visualize our graph
# graph = workflow.compile(checkpointer=memory)
graph = workflow.compile()
