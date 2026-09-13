# eval_retriever.py
from dotenv import load_dotenv
import json 
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from custom_llm import GroqModel
load_dotenv()
model = GroqModel()

from src.rag_deepeval.rag_pipeline import RagPipeline
load_dotenv()


JUDGE_MODEL = "openai/gpt-oss-20b"
GOLDEN_PATH = r"goldens/trial_faithfulness_datatset.json"
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


rag = RagPipeline()

test_cases = []

for g in goldens:
    result = rag.invoke(g['query'])

    test_cases.append(
        LLMTestCase(
            input = g['query'],
            retrieval_context = result["context"], 
            actual_output = result['answer']
                            )
    )

metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=model, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True),
    ContextualRelevancyMetric(threshold=THRESHOLD, model=model, include_reason=True)
]


evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters= {
            "retriever": "reranker",     ### if not using reranker then base_k5     
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 3,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
)




