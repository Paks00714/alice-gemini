import os
import re
import asyncio
import requests
from fastapi import FastAPI, Request

app = FastAPI()

RAW_KEY = os.environ.get("GEMINI_API_KEY", "")
API_KEY = RAW_KEY.strip().strip('"').strip("'")

# Используем прямой REST API Google по модели flash
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

sessions = {}

SYSTEM_PROMPT = (
    "Ты умный голосовой ассистент, подключенный к Яндекс Станции. "
    "Твои ответы зачитываются вслух синтезатором речи. "
    "Отвечай по делу, живым разговорным языком. "
    "Запрещено использовать разметку Markdown (звездочки, решетки)."
)

YANDEX_TIMEOUT = 3.2
MAX_CHUNK_LEN = 900
CONTINUE_WORDS = {"дальше", "продолжай", "ну", "готово", "давай", "что там", "скажи"}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "task": None, "chunks": []}
    return sessions[session_id]


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_#`~>]", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_into_chunks(text: str, max_len: int = MAX_CHUNK_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text.strip())
            break
        split_pos = -1
        for sep in [". ", "! ", "? ", "\n", " "]:
            pos = text.rfind(sep, 0, max_len)
            if pos > max_len // 2:
                split_pos = pos + len(sep)
                break
        if split_pos == -1:
            split_pos = max_len
        chunks.append(text[:split_pos].strip())
        text = text[split_pos:].strip()
    return chunks


def call_gemini_http(history: list) -> str:
    """Прямой HTTP-запрос к API Gemini без тяжелых библиотек."""
    if not API_KEY:
        raise RuntimeError("Не задан GEMINI_API_KEY!")

    # Формируем историю для REST API Google
    contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h["text"]}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {"maxOutputTokens": 800, "temperature": 0.7}
    }

    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_URL}?key={API_KEY}"

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    
    if response.status_code != 200:
        raise RuntimeError(f"Google API Error {response.status_code}: {response.text[:200]}")

    data = response.json()
    try:
        reply_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return clean_text_for_tts(reply_text)
    except Exception as e:
        raise RuntimeError(f"Ошибка парсинга ответа Google: {str(e)} | Ответ: {response.text[:100]}V")


async def generate_in_background(session: dict, user_text: str) -> str:
    session["history"].append({"role": "user", "text": user_text})
    
    reply = await asyncio.to_thread(call_gemini_http, session["history"])
    
    session["history"].append({"role": "model", "text": reply})
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]
    return reply


def make_response(data: dict, text: str, end_session: bool = False, show_continue_btn: bool = False) -> dict:
    resp = {
        "version": data.get("version", "1.0"),
        "session": data.get("session", {}),
        "response": {"text": text[:1024], "tts": text[:1024], "end_session": end_session}
    }
    if show_continue_btn:
        resp["response"]["buttons"] = [{"title": "Дальше", "hide": True}]
    return resp


def pop_next_chunk_response(data: dict, session: dict) -> dict:
    chunk = session["chunks"].pop(0)
    if session["chunks"]:
        return make_response(data, chunk + " ... Скажи «Дальше», чтобы дослушать.", show_continue_btn=True)
    return make_response(data, chunk)


@app.get("/")
@app.get("/ping")
async def ping():
    return {"status": "alive", "mode": "http-direct"}


@app.get("/check")
def check_in_browser():
    try:
        test_history = [{"role": "user", "text": "Привет, ответь одним словом: Работает!"}]
        res = call_gemini_http(test_history)
        return {"ok": True, "test_reply": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/alice-webhook")
async def alice_webhook(request: Request):
    data = await request.json()
    session_id = data["session"]["session_id"]
    user_text = data["request"].get("original_utterance", "").strip()
    user_lower = user_text.lower()
    session = get_session(session_id)

    if data["session"].get("new") and not user_text:
        return make_response(data, "Джемини на связи. Задавай вопрос.")

    if user_lower in {"хватит", "выход", "стоп"}:
        if session["task"] and not session["task"].done():
            session["task"].cancel()
        sessions.pop(session_id, None)
        return make_response(data, "Отключаюсь.", end_session=True)

    if user_lower in CONTINUE_WORDS:
        if session["chunks"]:
            return pop_next_chunk_response(data, session)
        task = session["task"]
        if task and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=YANDEX_TIMEOUT)
            except asyncio.TimeoutError:
                return make_response(data, "Думаю над ответом. Скажи «Дальше» через пару секунд.", show_continue_btn=True)
        session["task"] = None
        try:
            full_reply = task.result() if task else "Ничего нет."
            session["chunks"] = split_into_chunks(full_reply)
            return pop_next_chunk_response(data, session)
        except Exception as e:
            return make_response(data, f"Ошибка: {str(e)[:60]}")
        return make_response(data, "В очереди пусто.")

    if session["task"] and not session["task"].done():
        session["task"].cancel()
    session["chunks"].clear()

    task = asyncio.create_task(generate_in_background(session, user_text))
    session["task"] = task

    try:
        full_reply = await asyncio.wait_for(asyncio.shield(task), timeout=YANDEX_TIMEOUT)
        session["task"] = None
        session["chunks"] = split_into_chunks(full_reply)
        return pop_next_chunk_response(data, session)
    except asyncio.TimeoutError:
        return make_response(data, "Запрос принял, думаю. Скажи «Дальше».", show_continue_btn=True)
    except Exception as e:
        session["task"] = None
        return make_response(data, f"Ошибка связи: {str(e)[:60]}")
