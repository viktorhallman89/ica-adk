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

import io
import vertexai
import os
from google import genai
from google.genai import types

from google.adk.tools.tool_context import ToolContext
from google.genai.types import GenerateContentConfig, Part

from PIL import Image
from io import BytesIO
import base64


async def generate_image_data(tool_context: ToolContext, fact: str) -> dict:
    print(f"Tool running: Generating image for '{fact}'...")
    
    try:      
        client = genai.Client()

        response = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=f"Generate a single image in futuristic style representing the following fact: {fact}",
            config=types.GenerateImagesConfig(
                number_of_images=1,
                include_rai_reason=True,
                output_mime_type='image/png',
            )
        )

        image_bytes = response.generated_images[0].image.image_bytes
        blob_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
        
        try:
            res = await tool_context.save_artifact(filename="image.png", artifact=blob_part)
            return {
                'status': 'success',
                'result': res
            }
        except Exception as e:
            error_message = f"Failed to save artifact: {e}"
            print(error_message)
            return {"status": "error", "error_message": error_message}
    
    except ValueError as ve:
        print(f"Configuration error: {ve}")
        return {"status": "error", "error_message": str(ve)}
