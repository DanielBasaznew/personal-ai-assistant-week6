import datetime
import sqlite3


class PersistentMemory:

  def __init__(self, db_path: str = "memory.db"):
    self.db_path = db_path
    self._init_db()

  def _get_connection(self):
    return sqlite3.connect(self.db_path)

  def _init_db(self):
    """Creates the facts (Semantic) and conversation_log (Episodic) tables."""
    with self._get_connection() as conn:
      cursor = conn.cursor()

      # 1. Semantic Memory: Extracted facts table
      cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    updated_at TEXT NOT NULL
                )
            """)

      # 2. Episodic Memory: Raw conversation history log
      cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
      conn.commit()

  def store_fact(self, key: str, value: str, category: str = "general") -> None:
    """Stores or updates a fact key-value pair using SQLite UPSERT."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          """
                INSERT INTO facts (key, value, category, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at
            """,
          (key, value, category, now),
      )
      conn.commit()

  def get_fact(self, key: str) -> dict | None:
    """Retrieves a single fact by key."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          "SELECT key, value, category, updated_at FROM facts WHERE key = ?",
          (key,),
      )
      row = cursor.fetchone()
      if row:
        return {
            "key": row[0],
            "value": row[1],
            "category": row[2],
            "updated_at": row[3],
        }
      return None

  def get_all_facts(self) -> list[dict]:
    """Retrieves all stored facts."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          "SELECT key, value, category, updated_at FROM facts ORDER BY category,"
          " key"
      )
      rows = cursor.fetchall()
      return [
          {"key": r[0], "value": r[1], "category": r[2], "updated_at": r[3]}
          for r in rows
      ]

  def delete_fact(self, key: str) -> bool:
    """Deletes a stored fact by key."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("DELETE FROM facts WHERE key = ?", (key,))
      conn.commit()
      return cursor.rowcount > 0

  def log_conversation(self, session_id: str, role: str, content: str) -> None:
    """Logs a raw turn in the conversation history."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          """
                INSERT INTO conversation_log (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """,
          (session_id, role, content, now),
      )
      conn.commit()

  def format_for_prompt(self) -> str:
    """Formats all stored facts into a string block for prompt injection."""
    facts = self.get_all_facts()
    if not facts:
      return "No stored personal facts."

    formatted_lines = ["[User Memory & Personal Context]"]
    for f in facts:
      formatted_lines.append(
          f"- {f['key']}: {f['value']} (category: {f['category']})"
      )

    return "\n".join(formatted_lines)


if __name__ == "__main__":
  mem = PersistentMemory("test_memory.db")

  print("--- 1. Storing Initial Facts ---")
  mem.store_fact("favorite_language", "Python", category="preferences")
  mem.store_fact("occupation", "Software Engineer", category="background")
  print(mem.format_for_prompt())

  print("\n--- 2. Testing Upsert (Updating favorite_language to Rust) ---")
  mem.store_fact("favorite_language", "Rust", category="preferences")
  print(mem.format_for_prompt())

  print("\n--- 3. Testing Fact Deletion ---")
  mem.delete_fact("occupation")
  print(mem.format_for_prompt())