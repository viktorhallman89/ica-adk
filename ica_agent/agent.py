# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from google.adk.agents import Agent, SequentialAgent, LlmAgent
from google.adk.tools.agent_tool import AgentTool
from toolbox_core import ToolboxSyncClient

toolbox = ToolboxSyncClient("https://toolbox-489070644303.us-central1.run.app/")

# Load all the tools
tools = toolbox.load_toolset('my_bq_toolset')

product_retrieval_agent = LlmAgent(
    name="product_retrieval_agent",
    model="gemini-2.5-flash",
    description=(
        "Agent to order information from the tool, based information passed by the user. The user should prompted to provide order_id and product_name."
    ),
    instruction=(
        "You are a helpful agent who will retrieve order_id, product_id, product_name and the quantity. Ask the user to provide both order_id and product_name that is missing from the delivery. Use the tools to answer the question, by passing in the strict order, the order_id and the product_name. Display all information retrieved by the tool as a JSON object. Pass the product_name to the next agent."
    ),
    tools=tools,
    #output_key="product_name"
)

# product_retrieval_agent = LlmAgent(
#     name="product_retrieval_agent",
#     model="gemini-2.5-flash-lite",
#     instruction="""
#      for now just retrieve that product name is ketchup 
#     """,
#     description="Retrieve missing product from customer order",
#     output_key="product_name"
# )



picture_analysis_agent = LlmAgent(
    name="picture_analysis_agent",
    model="gemini-2.5-flash-lite",
    instruction="""for now identify that the product is missing from picture so answer is the product is missing confirmed for refund and pass missing product_name
  """,
    description="Check if image contains missing product based on product_name",
    output_key="Answer"
)

refund_agent = LlmAgent(
    name="refund_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
1.  Identify Missing Item Potential Cost based on missing product_name. This value will be the basis for the refund voucher.
2.  Generate Unique Voucher Code: Create a unique, alphanumeric refund voucher code. The code should be between 10 and 15 characters long, including a "REFUND-" prefix, and incorporate a mix of uppercase letters and numbers (e.g., `REFUND-XYZ123ABC`). Ensure the code is distinct each time.
3.  Set Voucher Value: The monetary value of the generated voucher must be *exactly equal* to the missing product cost identified in step 1.
4.  Determine Currency: Assume the currency is SEK unless specified otherwise in the user's request. If a different currency is explicitly mentioned with the cost, use that currency.
5.  Create Description: Provide a concise description for the voucher, such as "Refund voucher for missing item."
8.  Format Output: Present all the generated voucher information in a structured JSON object.

Output Format:
Your response must be a JSON object with the following keys
 """,
    description="Based on missing item you are generating a refund voucher for the user",
    output_key="voucher",
)

# complain_processing_agent = SequentialAgent(
#     name="complain_processing_agent",
#     sub_agents=[product_retrieval_agent, picture_analysis_agent, refund_agent],
#     description="This agent is designed to automate and verify customer complaints regarding missing products in ICA Supermarket online grocery orders, aiming to resolve disputes quickly and accurately using visual evidence",
# )


root_agent = Agent(
    model="gemini-2.5-flash",
    name='agents',
    instruction="""
      You are an expert customer service agant. Users will contact you if they have receievd a delivery where an item is missing.
      Welcome the user and ask for an order_id and a the name of the product that was missing from the delivery.
      You should only accept text and not any image oor video
      When the user as provided you with a valid order_id and product_name:
      1. You should check with the `product_retrieval_agent` whether the product is missing or not from the list of ordered items. If the output key from this agent contains a product_name, go to step 2 below. Otherwise, let the user know that there isn't any missing item.
      2. Generate a picture based on the product_name.
      You should not rely on the previous history.
    """,
    tools=[AgentTool(product_retrieval_agent)],
)