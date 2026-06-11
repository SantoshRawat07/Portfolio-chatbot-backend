import json
import os

MEMORY_FILE = "chat_memory/memory.json"


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return []

    with open(
        MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_memory(memory):

    os.makedirs(
        "chat_memory",
        exist_ok=True
    )

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            memory,
            f,
            indent=2,
            ensure_ascii=False
        )


def add_message(role, content):

    memory = load_memory()

    memory.append({
        "role": role,
        "content": content
    })

    memory = memory[-20:]

    save_memory(memory)