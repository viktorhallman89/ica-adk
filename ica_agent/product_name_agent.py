from google.adk.agents import Agent
from toolbox_core import ToolboxSyncClient

toolbox = ToolboxSyncClient("https://toolbox-489070644303.us-central1.run.app/")

# Load all the tools
tools = toolbox.load_toolset('my_bq_toolset')

root_agent = Agent(
    name="agents",
    model="gemini-2.5-flash",
    description=(
        "Agent to order information from the tool, based information passed by the user. The user should prompted to provide order_id and product_name."
    ),
    instruction=(
        "You are a helpful agent who will retrieve order_id, product_id, product_name and the quantity. Ask the user to provide both order_id and product_name that is missing from the delivery. Use the tools to answer the question, by passing in the strict order, the order_id and the product_name. Display all information retrieved by the tool as a JSON object. Print the query passed to the tools"
    ),
    tools=tools,
)