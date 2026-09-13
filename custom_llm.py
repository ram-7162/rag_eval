# from deepeval.models.base_model import DeepEvalBaseLLM
# import os
# from dotenv import load_dotenv
# load_dotenv()
# from groq import Groq
# # DeepEval doesn't need to understand every provider's API.
# # It just says:
# # "Give me an LLM object that follows my interface."
# # Your job is to adapt Groq's API to DeepEval's interface.
# # This is why this class is often called a wrapper or adapter.

# # → DeepEval stores your object
# # → DeepEval needs an LLM
# # → DeepEval calls the interface methods
# # → your GroqModel method executes
# # → that method calls Groq
# # → Groq returns the judge's response
# # → DeepEval uses that response to calculate the metric.

# class GroqModel(DeepEvalBaseLLM):

#     def __init__(self, model="openai/gpt-oss-20b"):
#         self.model = model
#         self.client = Groq(api_key=os.getenv("GROQ_API_KEY2"))

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str, schema=None):

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0
#         )

#         return response.choices[0].message.content

#     async def a_generate(self, prompt: str, schema=None):

#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0
#         )

#         return response.choices[0].message.content

#     def get_model_name(self):
#         return self.model




import os
import json
from groq import Groq, AsyncGroq
from deepeval.models import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model="openai/gpt-oss-20b", max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY2"))
        self.async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY2"))

    def load_model(self):
        return self.client

    def _build_messages(self, prompt: str, schema=None):
        """Inject JSON-mode instructions + the target schema into the prompt."""
        if schema is not None:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            prompt = (
                f"{prompt}\n\n"
                "Respond with ONLY a single valid JSON object — no markdown "
                "fences, no commentary before or after it — that matches "
                f"this JSON schema exactly:\n{schema_json}"
            )
        return [{"role": "user", "content": prompt}]

    def _parse(self, raw: str, schema):
        """Parse+validate raw model output against the schema, or raise."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # strip ```json ... ``` fences if the model added them anyway
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
        if schema is not None:
            validated = schema.model_validate(data)
            return validated.model_dump_json()
        return json.dumps(data)

    def generate(self, prompt: str, schema=None):
        messages = self._build_messages(prompt, schema)
        last_err = None
        for attempt in range(self.max_retries):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            try:
                return self._parse(raw, schema)
            except Exception as e:
                last_err = e
                continue
        raise ValueError(
            f"GroqModel failed to produce valid JSON after "
            f"{self.max_retries} attempts: {last_err}"
        )

    async def a_generate(self, prompt: str, schema=None):
        messages = self._build_messages(prompt, schema)
        last_err = None
        for attempt in range(self.max_retries):
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            try:
                return self._parse(raw, schema)
            except Exception as e:
                last_err = e
                continue
        raise ValueError(
            f"GroqModel failed to produce valid JSON after "
            f"{self.max_retries} attempts: {last_err}"
        )

    def get_model_name(self):
        return self.model