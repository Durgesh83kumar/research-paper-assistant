import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.factory import get_llm

# ---- PDF Q&A Evaluator ----
EVAL_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "context"],
    template="""You are an expert evaluator for AI answers. 
Evaluate the following answer on 3 metrics and respond ONLY with a JSON object.

Question: {question}
Answer: {answer}
Context from paper: {context}

Rate each metric from 0 to 100:

1. Relevance: Does the answer directly address the question?
2. Faithfulness: Is the answer based on the provided context? (not hallucinated)
3. Completeness: Does the answer fully cover the question or just partially?

Respond ONLY with this exact JSON format, nothing else:
{{
    "relevance": <score 0-100>,
    "faithfulness": <score 0-100>,
    "completeness": <score 0-100>,
    "reasoning": "<one sentence explaining the scores>"
}}"""
)


def evaluate_answer(question: str, answer: str, context: str) -> dict:
    """Evaluate a PDF-based answer using LLM as a judge."""
    try:
        llm = get_llm(use_case="eval")
        parser = StrOutputParser()
        chain = EVAL_PROMPT | llm | parser

        raw = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context,
        })

        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)
        return {
            "relevance": int(scores.get("relevance", 0)),
            "faithfulness": int(scores.get("faithfulness", 0)),
            "completeness": int(scores.get("completeness", 0)),
            "reasoning": scores.get("reasoning", ""),
            "error": None,
            "type": "pdf",
        }

    except Exception as e:
        return {
            "relevance": 0,
            "faithfulness": 0,
            "completeness": 0,
            "reasoning": "Evaluation failed",
            "error": str(e),
            "type": "pdf",
        }


# ---- General Chat Evaluator ----
GENERAL_EVAL_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an expert evaluator for AI answers.
Evaluate the following answer and respond ONLY with a JSON object.

Question: {question}
Answer: {answer}

Rate each metric from 0 to 100:
1. Relevance: Does the answer directly address the question?
2. Clarity: Is the answer clear and easy to understand?
3. Completeness: Does the answer fully cover the question?

Respond ONLY with this exact JSON format, nothing else:
{{
    "relevance": <score 0-100>,
    "clarity": <score 0-100>,
    "completeness": <score 0-100>,
    "reasoning": "<one sentence explaining the scores>"
}}"""
)


def evaluate_general_answer(question: str, answer: str) -> dict:
    """Evaluate a general (non-PDF) answer using LLM as a judge."""
    try:
        llm = get_llm(use_case="eval")
        parser = StrOutputParser()
        chain = GENERAL_EVAL_PROMPT | llm | parser

        raw = chain.invoke({
            "question": question,
            "answer": answer,
        })

        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)
        return {
            "relevance": int(scores.get("relevance", 0)),
            "clarity": int(scores.get("clarity", 0)),
            "completeness": int(scores.get("completeness", 0)),
            "reasoning": scores.get("reasoning", ""),
            "error": None,
            "type": "general",
        }

    except Exception as e:
        return {
            "relevance": 0,
            "clarity": 0,
            "completeness": 0,
            "reasoning": "Evaluation failed",
            "error": str(e),
            "type": "general",
        }


# ---- Score Renderer ----
def render_scores(scores: dict):
    """Returns HTML string for score display."""
    def bar(score):
        color = "#4CAF50" if score >= 70 else "#FF9800" if score >= 40 else "#F44336"
        return (
            f'<div style="background:#333; border-radius:10px; height:12px; margin:3px 0;">'
            f'<div style="background:{color}; width:{score}%; height:12px; '
            f'border-radius:10px;"></div></div>'
        )

    is_general = scores.get("type") == "general"

    if is_general:
        metric2_label = "Clarity"
        metric2_value = scores.get("clarity", 0)
    else:
        metric2_label = "Faithfulness"
        metric2_value = scores.get("faithfulness", 0)

    html = f"""
    <div style="background:#1e1e1e; padding:12px; border-radius:10px; margin-top:10px;
                border-left: 3px solid #4CAF50;">
        <p style="color:#aaa; margin:0 0 8px 0; font-size:13px;">📊 <b>Answer Quality</b></p>
        <p style="color:#fff; margin:2px 0; font-size:13px;">
            Relevance &nbsp;&nbsp;&nbsp;&nbsp; {scores['relevance']}%</p>
        {bar(scores['relevance'])}
        <p style="color:#fff; margin:6px 0 2px 0; font-size:13px;">
            {metric2_label} &nbsp; {metric2_value}%</p>
        {bar(metric2_value)}
        <p style="color:#fff; margin:6px 0 2px 0; font-size:13px;">
            Completeness {scores['completeness']}%</p>
        {bar(scores['completeness'])}
        <p style="color:#aaa; margin:8px 0 0 0; font-size:12px;">
            💡 {scores['reasoning']}</p>
    </div>
    """
    return html