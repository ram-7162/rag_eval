# eval_retriever.py
from dotenv import load_dotenv
import json 
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric


from src.rag_deepeval.retriever import build_retriever
from src.rag_deepeval.reranker import RerankingRetriever

load_dotenv()
from custom_llm import GroqModel
load_dotenv()
model = GroqModel()

JUDGE_MODEL = "openai/gpt-oss-20b"
GOLDEN_PATH = "goldens/trial_retriever_goldens.json"
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# retriever = build_retriever()
retriever = RerankingRetriever()

test_cases = []

for g in goldens:
    retrieved = retriever.invoke(g['query'])
    retrieval_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input = g['query'],
            expected_output = g['ideal_answer'],
            retrieval_context = retrieval_context, 
            actual_output = "(generator not evaluated in this run)"
        )
    )

metrics = [
    ContextualPrecisionMetric(threshold=THRESHOLD, model=model, include_reason=True),
    ContextualRecallMetric(threshold=THRESHOLD, model=model, include_reason=True)
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




