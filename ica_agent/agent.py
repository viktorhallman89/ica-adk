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

from google.adk.agents import Agent

# Import the tools
from .tools import generate_image_data
from google.adk.tools.load_artifacts_tool import load_artifacts_tool



root_agent = Agent(
    model="gemini-2.0-flash-001",
    name='agents',
    instruction="""
      Ask the user for a year in arabic format.
      You can only accept a year up until the current date's year.
      You can only accept integers, and should refuse any inputs such as text or media.
      When the user as provided you with a valid year:
      1. You should respond with 1 facts from the year provided by the user
      2. Based on this fact from the previous step, call the `generate_image_data` and display the image_artifact in the response.
      You should not rely on the previous history.
    """,
    tools=[generate_image_data, load_artifacts_tool],
)
