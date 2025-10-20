from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class OpenAIWrapper:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=api_key)
    
    async def generate_chat_completion(self, system_prompt, user_prompt, temperature=1):
        response = await self.openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    async def generate_chat_completion_structured_output(self, system_prompt, user_prompt, resp_format):

        response = await self.openai_client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=resp_format
        )
        return response

    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> list:
        response = await self.openai_client.embeddings.create(
                input= text,
                model= model,
        )
        embedding = response.data[0].embedding
        return embedding

    async def moderate_text(self, text):
        response = await self.openai_client.moderations.create(
            model="text-moderation-latest",
            input=text
        )

        return response.results[0].flagged