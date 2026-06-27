import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(use_case: str = "chat"):
    """
    use_case: 
        "chat" → best model for Q&A
        "summary" → faster model for summarization
        "eval" → faster model for evaluation
    """
    provider = os.getenv("LLM_PROVIDER", "groq")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        if use_case == "chat":
            model = "llama-3.3-70b-versatile"  # best for Q&A
        else:
            model = "llama-3.1-8b-instant"  # fast for summary/eval

        return ChatGroq(
            model=model,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")