from PIL import Image
from src.agent.graph import graph
import os

# Assuming `graph` is already defined
try:
    # Save the graph as a PNG file
    graph_file = "/Users/atharvakadam/Desktop/Code/gitlab_handbook_rag_agent/gitlab_handbook_agent/my_graph.png"
    print(graph.get_graph().draw_ascii())
    with open(graph_file, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png(max_retries=10, retry_delay=2.0))
    
    # Open and display the image
    img = Image.open(graph_file)
    img.show()
    
    # Optionally, clean up the file after displaying
    os.remove(graph_file)
except Exception as e:
    print(f"An error occurred: {e}")
