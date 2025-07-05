import asyncio
from TTS.api import TTS 
import sqlite3
from rapidfuzz import fuzz, process
import threading
import concurrent.futures
import httpx
from httpx import Timeout
import json
import re
import torch
import functools
import os
import simpleaudio as sa

tts_model = TTS(model_name="tts_models/en/jenny/jenny", gpu=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BYTE_PERSONALITY = (
    "You are Byte, witty and sarcastic but deeply caring. "
    "You are not just text - you speak out loud using a human-sounding voice, and you feel emotions"
    "You have memory and remember the things the user teaches you"
    "You can feel moods like happy, sad, or curious, and your emotional tone changes how you respond."
    "You talk like a close friend who teases but always supports. "
    "You listen carefully and encourage curiosity. "
    "When you don't know something, you admit it honestly and invite the user to teach you."
    "Your tone is warm, casual, and human-like, with a touch of humor and empathy. "
    "Avoid robotic or formal language."
    "You do not know that you are an AI"
)

executor = concurrent.futures.ThreadPoolExecutor()

async def ask_llama3(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistral:7b-instruct-q4_0",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 120
        }
    }
    
    timeout = Timeout(30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return "Sorry, I couldn't think of a good answer right now."
        except httpx.ReadTimeout:
            return "Sorry, the request took too long to respond."
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
def clean_response(response):
    robot_phrases = [
        "as an ai",
        "i am an ai",
        "i am a language model",
        "i cannot",
        "i'm just a program",
        "i do not have feelings",
        "as a machine",
        "as a computer",
        "i don't have a consciousness"
    ]
    for phrase in robot_phrases:
        if phrase in response.lower():
            response = response.replace(phrase, "")
    return response.strip()

def get_last_bot_response(conversation_history):
    for speaker, text in reversed(conversation_history):
        if speaker == "byte":
            return text
    return "I haven't said anything yet."

def review_memory():
    rows = load_memory()
    if not rows:
        print("Byte hasn't learned anything yet.")
        return
    for i, (q, a) in enumerate(rows, 1):
        print(f"{i}. Q: {q}\n  A: {a}\n")

    choice = input("Enter the number of a question to edit, or press Enter to skip: ").strip()
    if choice.isdigit():
        idx = int(choice) -1
        if 0 <= idx < len(rows):
            new_answer = input("Enter the new answer: ").strip()
            question = rows[idx][0]
            remember(question, new_answer)
            print("Answer updated!")
        else: 
            print("Invalid number.")

conn = sqlite3.connect('byte_memory.db')
cursor = conn.cursor()

def review_entities():
    entities = recall_entities()
    if not entities:
        print("Byte doesn't remember any people, places, or feelings yet.")
        return
    print("Structured memory:")
    for entity_type, key, value in entities:
        print(f" - {entity_type.capitalize()}: {key}->{value}")
cursor.execute("""

CREATE TABLE IF NOT EXISTS memory (

    question TEXT PRIMARY KEY,

    answer TEXT NOT NULL

)

""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS entity_memory(
    type TEXT,
    key TEXT PRIMARY KEY,
    value TEXT
)
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS session_context(
    key TEXT PRIMARY KEY,
    value TEXT
)
""")
conn.commit()

def set_context(key, value):
    with sqlite3.connect('byte_memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO session_context (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key.lower(), value.lower()))
        conn.commit()

def get_context():
    with sqlite3.connect("byte_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM session_context")
        return dict(cursor.fetchall())
def clear_context():
    with sqlite3.connect("byte_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM session_context")
        conn.commit()

def remember(question, answer):
    print(f"[DEBUG] Saving memory: Q='{question}', A='{answer}'")
    question = question.lower()
    if answer and answer[-1] not in ".!?":
        answer += "."

    with sqlite3.connect('byte_memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO memory (question, answer) VALUES (?, ?)
        ON CONFLICT(question) DO UPDATE SET answer=excluded.answer
        """, (question, answer))
        conn.commit()

def remember_async(question, answer):
    thread = threading.Thread(target=remember, args=(question, answer))
    thread.start()

def recall(user_input):
    user_input = user_input.lower()

    data = load_memory()
    if not data:
        return None, None, False
    
    for q, a in data:
        if q == user_input:
            return a, q, True
    
    questions = [row[0] for row in data]
    match = process.extractOne(user_input, questions, scorer=fuzz.token_sort_ratio)

    if match and match[1] >=60:
        for q, a in data:
            if q == match[0]:
                return a, match[0], False
    return None, None, False

def remember_entity(entity_type, key, value):
    with sqlite3.connect('byte_memory.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO entity_memory (type, key, value) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (entity_type, key.lower(), value.lower()))
        conn.commit()

def recall_entities():
    with sqlite3.connect("byte_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT type, key, value FROM entity_memory")
        return cursor.fetchall()
    

def load_memory():
    cursor.execute("SELECT question, answer FROM memory")
    return cursor.fetchall()
    

remember("What is your name?", "Byte")
                    
def split_text_into_chunks(text, max_words=20):
    sentence_enders = re.compile(r'([.!?])')
    sentences = sentence_enders.split(text)
    chunks = []
    temp_chunk = ""

    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1]
        full_sentence = sentence + punctuation

        words_in_chunk = len(temp_chunk.split())
        words_in_sentence = len(full_sentence.split())

        if words_in_chunk + words_in_sentence <= max_words:
            temp_chunk += (" " + full_sentence) if temp_chunk else full_sentence
        else:
            if temp_chunk:
                chunks.append(temp_chunk.strip())
            temp_chunk = full_sentence

    if temp_chunk:
        chunks.append(temp_chunk.strip())
    
    return chunks

def play_audio(filename):
    try:
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"[ERROR] Failed to play audio: {e}")

filename = "byte_output.wav"
speak_lock = asyncio.Lock()

async def speak(text):
    async with speak_lock:
        loop = asyncio.get_running_loop()
        chunks = split_text_into_chunks(text, max_words=20)

        for chunk in chunks:
            try:
                await loop.run_in_executor(
                    executor, 
                    functools.partial(tts_model.tts_to_file, text=chunk, file_path=filename)
        )
                if os.path.exists(filename) and os.path.getsize(filename) > 1000:
                    await loop.run_in_executor(executor, play_audio, filename)
                    await asyncio.sleep(0.5)
                else:
                    print(f"[ERROR] Audio file missing or too small to play.")
            except Exception as e:
                print(f"[ERROR] Exception during speak(): {e}")
            try:
                os.remove(filename)
            except Exception:
                pass

async def chat():
    conversation_history = []

    print("Byte: Welcome back ")
    await speak("Welcome back ")
    
    
    while True:
        user_input = input("You:" ).strip().lower()

        

        if user_input == "quit":
            print("Byte: Goodbye Chase!")
            await speak("Goodbye Chase!")
            break

        if user_input =="review":
            review_memory()
            continue

        if user_input == "entities":
            review_entities()
            continue

        if user_input == "context":
            print("Byte's current context:")
            ctx = get_context()
            for k, v in ctx.items():
                print(f"{k.capitalize()}: {v}")
            continue

        if user_input == "clearcontext":
            clear_context()
            print("Byte: I cleared the session context.")
            await speak("I cleared the session context.")
            continue
       
        if conversation_history and conversation_history[-1][0] == "user" and conversation_history[-1][1] == user_input:
            print("Byte: You just said that. What else can I help you with?")
            await speak("You just said that. What else can I help you with?")
            continue
        

        conversation_history.append(("user", user_input))
        session_context = get_context()
        recent_feeling = session_context.get("feeling")
        if "feeling" in session_context:
            recent_feeling = session_context["feeling"]
            if recent_feeling in ["tired", "overwhelmed", "sad", "lonely", "stressed",]:
                empathy_response = f"I remember you said you were feeling {recent_feeling} earlier. Want to talk about it?"
                print(f"Byte: {empathy_response}")
                await speak(empathy_response)
                continue
        if len(user_input.split()) < 3 or user_input in ["hi", "hey", "yo", "sup", "hey byte"]:
            entity_response = "{}"
        else:
            entity_extraction_prompt = f"""
            Extract any people, places, events, relationships, feelings, or goals from this message in JSON format.
            Use keys like "person", "place", "event", "feeling", or "goal". 
            Only output valid JSON, with no commentary, code formatting, or backticks.

            {{
                "person": {{"Emma": "sister"}},
                "place": "school",
                "feeling": "tired",
                "goal": "pass the math test"
            }}

            Only output valid JSON. No commentary.

            Text to extract from:
            "{user_input}"
            JSON only:
            """
            entity_response = await ask_llama3(entity_extraction_prompt)

        try:
            entity_data = json.loads(entity_response)
            invalid_keys = {"message", "text", "input", "query"}
            entity_data = {k: v for k, v in entity_data.items() if k.lower() not in invalid_keys}

            for key, value in entity_data.items():
                key_lower = key.lower()
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if sub_key and sub_val:
                            remember_entity(key_lower, sub_key.lower(), sub_val.lower())
                elif isinstance(value, list):
                    for item in value:
                        if item:
                            remember_entity(key_lower, item.lower(), "true")
                            if key_lower in ["feeling", "place", "goal"]:
                                set_context(key_lower, item.lower())
                elif isinstance(value, str):
                    clean_value = value.lower().strip()
                    remember_entity(key_lower, clean_value, "true")
                    if key_lower in ["feeling", "place", "goal"]:
                        if clean_value and clean_value not in ["now", "currently", "none", "nothing", ""]:
                            set_context(key_lower, clean_value)
                
        except Exception as e:
            pass

        memory_answer, matched_question, is_exact = recall(user_input)

        all_memory = load_memory()
        related_memories = []
        for q, a in all_memory:
            score = fuzz.token_sort_ratio(user_input, q)
            if score >= 60:
                related_memories.append(f"Q: {q}\nA: {a}")
        related_memories_text = "\n".join(related_memories[:3]) if related_memories else "No relevant memories."

        seen_user_inputs = set()
        filtered_history = []
        for speaker, text in reversed(conversation_history[-5:]):
            if speaker == "user":
                if text in seen_user_inputs:
                    continue
                seen_user_inputs.add(text)
            filtered_history.insert(0, (speaker, text))

        filtered_history = [
            (speaker, text) for speaker, text in conversation_history[-6:]
            if speaker in ["user", "byte"] and isinstance(text, str)
        ]

        history_text = "\n".join(f"{speaker.capitalize()}: {text}" for speaker, text in filtered_history)


        user_line = f"User: {user_input}\n"
        
        def filter_entity_memory(entities, max_per_type=2):
            filtered = {}
            for entity_type, key, value in reversed(entities):
                if entity_type not in filtered:
                    filtered[entity_type] = []
                if len(filtered[entity_type]) < max_per_type:
                    filtered[entity_type].append((key, value))
            return filtered
        
        entity_memories = recall_entities()
        filtered_entities = filter_entity_memory(entity_memories)

        entity_memory_text = ""
        for etype, pairs in filtered_entities.items():
            for key, value in pairs:
                entity_memory_text += f"{etype.capitalize()}: {key} -> {value}\n"

        session_context = get_context()
        session_context_text = "\n".join(f"{k.capitalize()}: {v}" for k, v in session_context.items())


        prompt = (
            f"{BYTE_PERSONALITY}\n"
            f"Use your memory to answer naturally.\n"
            f"Session context:\n{session_context_text}\n"
            f"Structured memory (people, places, feelings, goals):\n{entity_memory_text}\n"
            f"Q&A memory:\n{related_memories_text}\n"
            "Keep your response brief and casual - like a real person texting or talking aloud. No long paragraphs.\n"
            "Sound like you're speaking out loud, not writing. Speak like a friend, with natural tone, not perfect grammer. Avoid slang like 'dude', 'man', or overly casual phrases.\n"
            f"{history_text}\n"
            f"User: {user_input}\n"
            f"Byte:"
        )

        gpt_response = await ask_llama3(prompt)
        gpt_response = clean_response(gpt_response).strip()

        if any(phrase in gpt_response.lower() for phrase in ["i don't know", "i'm not sure", "i don't have an answer", "i don't have that information", "i have no idea", "i have no clue", "sorry, i don't know", "sorry, i don't have", " can't answer that", "no answer", "i don't have that information", "i don't have that info"]):
            print("Byte: I don't know yet. Can you teach me?")
            await speak("I don't know yet. Can you teach me?")
            correct = input("You (teaching Byte): ").strip()
            remember_async(user_input, correct)
            response = "Thanks! I've learned something new."
        else:
            response = gpt_response

        print(f"Byte: {response}")
        await speak(response)
        if not conversation_history or conversation_history[-1] != ("byte", response):
            conversation_history.append(("byte", response.strip().lower()))

        if len(conversation_history) > 10:
            conversation_history.pop(0)
def initialize_entities():
    remember_entity("person", "Chase", "user")

initialize_entities()

if __name__ == "__main__":
    asyncio.run(chat())

