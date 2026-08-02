import pandas as pd
from nodes import retrieve_node, route_node, output_node

# Load one real message
msg = pd.read_csv("../dataset/messages.csv").iloc[0].to_dict()
print("Message:", msg["message_id"], "|", msg["message_text"][:60])

# Build fake state
state = {"message": msg, "context": {}, "prediction": {}, "result": {}}

# Test retrieve_node
state = retrieve_node(state)
print("\n--- retrieve_node OK ---")
print("user:", state["context"]["user"])
print("business:", state["context"]["business"])

# Test route_node (makes real LLM call)
state = route_node(state)
print("\n--- route_node OK ---")
print("prediction:", state["prediction"])

# Test output_node
state = output_node(state)
print("\n--- output_node OK ---")
print("result:", state["result"])