from deepeval.models.base_model import DeepEvalBaseLLM
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
# DeepEval doesn't need to understand every provider's API.
# It just says:
# "Give me an LLM object that follows my interface."
# Your job is to adapt Groq's API to DeepEval's interface.
# This is why this class is often called a wrapper or adapter.

# → DeepEval stores your object
# → DeepEval needs an LLM
# → DeepEval calls the interface methods
# → your GroqModel method executes
# → that method calls Groq
# → Groq returns the judge's response
# → DeepEval uses that response to calculate the metric.

class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY2"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def get_model_name(self):
        return self.model