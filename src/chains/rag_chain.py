from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.factory import get_llm
from src.vectorstore.faiss_store import get_retriever

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "history", "question"],
    template="""You are a helpful research assistant chatting with a user.

You have access to content from a research paper the user uploaded (Context below).
Use it when the question relates to the paper. If the question is general/unrelated 
to the paper, answer it normally using your own knowledge.

Conversation so far:
{history}

Paper Context:
{context}

User Question: {question}

Answer (be conversational, mention page numbers only if you used the paper context):"""
)

GENERAL_PROMPT = PromptTemplate(
    input_variables=["history", "question"],
    template="""You are a helpful AI assistant chatting with a user.

Conversation so far:
{history}

User Question: {question}

Answer:"""
)

def _format_history(chat_history: list, max_turns: int = 20) -> str:
    recent = chat_history[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines) if lines else "(no previous conversation)"

def answer_question(question: str, chat_history: list, pdf_processed: bool, chat_id: str) -> dict:
    llm = get_llm()
    parser = StrOutputParser()
    history_text = _format_history(chat_history)

    if pdf_processed:
        retriever = get_retriever(chat_id)  # ← uses THIS chat's FAISS index
        source_documents = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in source_documents])

        chain = RAG_PROMPT | llm | parser
        answer = chain.invoke({
            "context": context,
            "history": history_text,
            "question": question,
        })
        return {"answer": answer, "sources": source_documents}

    else:
        chain = GENERAL_PROMPT | llm | parser
        answer = chain.invoke({
            "history": history_text,
            "question": question,
        })
        return {"answer": answer, "sources": []}