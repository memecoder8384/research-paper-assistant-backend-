import threading
from typing import Dict, List


class SessionStore:
    """Thread-safe in-memory session store for multi-turn research conversations."""

    _store: Dict[str, List[Dict[str, str]]] = {}
    _lock = threading.Lock()
    _max_turns: int = 10  # Maximum messages (5 user-assistant turns) per session

    @classmethod
    def get_history(cls, conversation_id: str) -> List[Dict[str, str]]:
        """Retrieve recent message history for a conversation session."""
        with cls._lock:
            return list(cls._store.get(conversation_id, []))

    @classmethod
    def add_turn(cls, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """Append a user question and assistant answer to the conversation session."""
        with cls._lock:
            if conversation_id not in cls._store:
                cls._store[conversation_id] = []
            history = cls._store[conversation_id]
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            # Keep history within reasonable boundary to avoid unbounded context growth
            if len(history) > cls._max_turns:
                cls._store[conversation_id] = history[-cls._max_turns :]

    @classmethod
    def clear_history(cls, conversation_id: str) -> bool:
        """Remove conversation history for a given session."""
        with cls._lock:
            if conversation_id in cls._store:
                del cls._store[conversation_id]
                return True
            return False

    @classmethod
    def reset_all(cls) -> None:
        """Clear all stored sessions (useful for test isolation)."""
        with cls._lock:
            cls._store.clear()
