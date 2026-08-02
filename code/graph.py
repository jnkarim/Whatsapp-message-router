from langgraph.graph import StateGraph, END
from state import RouterState
from nodes import retrieve_node, image_node, voice_node, route_node, output_node



def media_router(state: RouterState) -> str:
    """
    Called after retrieve_node.
    Checks the original message's media_type and
    routes to the correct processing node.

    Returns a string that matches an edge name in the graph.
    """
    media_type = state["message"].get("media_type")

    if media_type == "image":
        return "image"
    elif media_type == "voice":
        return "voice"
    else:
        return "text"


# Build the graph

def build_graph():
    """
    Assembles the full LangGraph pipeline.
    Called once at startup in main.py.
    Returns a compiled runnable graph.
    """

    # Initialize graph with your state schema
    graph = StateGraph(RouterState)

    # Add all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("image",    image_node)
    graph.add_node("voice",    voice_node)
    graph.add_node("route",    route_node)
    graph.add_node("output",   output_node)

    # Entry point
    graph.set_entry_point("retrieve")

    # After retrieve: branch based on media type
    graph.add_conditional_edges(
        "retrieve",         # from this node
        media_router,       # call this function to decide
        {
            "image": "image",   # if returns "image" → go to image node
            "voice": "voice",   # if returns "voice" → go to voice node
            "text":  "route",   # if returns "text"  → skip to route node
        }
    )

    # After media nodes: always go to route
    graph.add_edge("image", "route")
    graph.add_edge("voice", "route")

    # After route: always go to output
    graph.add_edge("route", "output")

    # After output: done
    graph.add_edge("output", END)

    # Compile and return
    return graph.compile()