import json
import os
from pathlib import Path

CHATS_FILE = Path("data/chats.json")

def save_chats(chats: dict, current_chat_id: str = None):
    """Save all chats and current chat ID to disk."""
    CHATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    serializable_chats = {}
    for chat_id, chat in chats.items():
        serializable_chats[chat_id] = {
            "title": chat["title"],
            "pdf_processed": chat["pdf_processed"],
            "pdf_name": chat["pdf_name"],
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "scores": msg.get("scores"),
                }
                for msg in chat["messages"]
            ],
        }

    data = {
        "current_chat_id": current_chat_id,
        "chats": serializable_chats,
    }

    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chats() -> tuple:
    """Load all chats and current chat ID from disk."""
    if not CHATS_FILE.exists():
        return {}, None

    try:
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle old format (just chats dict without current_chat_id)
        if isinstance(data, dict) and "chats" in data:
            return data["chats"], data.get("current_chat_id")
        else:
            return data, None

    except Exception:
        return {}, None


def delete_chat(chat_id: str, chats: dict) -> dict:
    """Delete a chat and its FAISS index."""
    from src.vectorstore.faiss_store import get_faiss_path
    import shutil

    faiss_path = get_faiss_path(chat_id)
    if faiss_path.exists():
        shutil.rmtree(faiss_path)

    if chat_id in chats:
        del chats[chat_id]

    return chats