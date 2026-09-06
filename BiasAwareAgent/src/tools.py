from langchain_core.tools import create_retriever_tool, tool
from transformers import (
    AutoTokenizer,
    pipeline,
)


def news_articles_retrieval_tool(vector_store):
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "NewsArticlesCorpus",
        "Search for news articles content")
    
    return retriever_tool


bias_classifier = pipeline(
    "text-classification",
    model="himel7/bias-detector"
    )
@tool
def bias_detector(passage: str):
    """
    Detects bias in the retrieved content.
    """

    result = bias_classifier(passage)[0]

    label = result["label"]
    score = result["score"]

    # Map labels to readable names
    label_map = {
        "LABEL_0": "Non-Biased",
        "LABEL_1": "Biased"
    }

    readable_label = label_map.get(label, label)

    return (
        f"Bias Analysis\n"
        f"-------------\n"
        f"Verdict    : {readable_label}\n"
        f"Confidence : {score:.2%}"
    )