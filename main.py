

import os
import re
import asyncio
from fastapi import FastAPI, Request
from google import genai
from google.genai import types

app = FastAPI()

# Берем API-ключ из переменных окружения Render (Environment Variables)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Не задана переменная окружения GEMINI_API_KEY!")

client = genai.Client(api_key=API_KEY)

# Хранилище сессий в оперативной памяти:
# history — контекст диалога с Gemini
# task — текущая фоновая задача генерации
# chunks — очередь кусков длинного ответа
sessions = {}

SYSTEM_PROMPT = (
    "Ты умный голосовой ассистент, подключенный к Яндекс Станции. "
    "Твои ответы зачитываются вслух синтезатором речи. "
    "Отвечай по делу, живым разговорным языком. "
    "Категорически запрещено использовать разметку Markdown (звездочки, решетки, жирный шрифт), "
    "сложные списки и куски кода, если об этом прямо не попросили."
)

# Таймаут перед ответом-заглушкой (лимит Яндекса ~4.5 сек с учетом пинга)
YANDEX_TIMEOUT = 3.2
# Максимальная длина одного куска текста (лимит Яндекса — 1024 символа)
MAX_CHUNK_LEN = 900

# Слова-триггеры для дочитывания ответа
CONTINUE_WORDS = {"дальше", "продолжай", "ну", "готово", "давай", "что там", "скажи"}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "task": None,
            "chunks": []
        }
    return sessions[session_id]


def clean_text_for_tts(text: str) -> str:
    """Очищает текст от Markdown-символов, чтобы колонка не читала их вслух."""
    text = re.sub(r"[*_#`~>]", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_into_chunks(text: str, max_len: int = MAX_CHUNK_LEN) -> list[str]:
    """Разбивает длинный ответ на части по границам предложений."""
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


def call_gemini_sync(history: list) -> str:
    """Синхронный запрос к Gemini API."""
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
            temperature=0.7
        )
    )
    return clean_text_for_tts(res.text or "Пустой ответ от модели.")


async def generate_in_background(session: dict, user_text: str) -> str:
    """Запускает генерацию в пуле потоков и обновляет историю диалога."""
    current_history = session["history"] + [
        types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
    ]

    reply = await asyncio.to_thread(call_gemini_sync, current_history)

    session["history"] = current_history + [
        types.Content(role="model", parts=[types.Part.from_text(text=reply)])
    ]
    # Храним последние 20 реплик (10 пар вопрос-ответ)
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    return reply


def make_response(data: dict, text: str, end_session: bool = False, show_continue_btn: bool = False) -> dict:
    """Формирует JSON-ответ по спецификации Яндекс Диалогов."""
    resp = {
        "version": data.get("version", "1.0"),
        "session": data.get("session", {}),
        "response": {
            "text": text[:1024],
            "tts": text[:1024],
            "end_session": end_session
        }
    }
    if show_continue_btn:
        resp["response"]["buttons"] = [{"title": "Дальше", "hide": True}]
    return resp


def pop_next_chunk_response(data: dict, session: dict) -> dict:
    """Отдает следующий кусок ответа из очереди."""
    chunk = session["chunks"].pop(0)
    if session["chunks"]:
        suffix = " ... Скажи «Дальше», чтобы дослушать."
        return make_response(data, chunk + suffix, show_continue_btn=True)
    return make_response(data, chunk)


@app.get("/")
@app.get("/ping")
async def ping():
    """Эндпоинт для проверки жизни сервера и защиты от сна (Cron-job / UptimeRobot)."""
    return {"status": "alive", "service": "alice-gemini-webhook"}


@app.post("/alice-webhook")
async def alice_webhook(request: Request):
    data = await request.json()
    session_id = data["session"]["session_id"]
    user_text = data["request"].get("original_utterance", "").strip()
    user_lower = user_text.lower()

    session = get_session(session_id)

    # 1. Приветствие при запуске навыка
    if data["session"].get("new") and not user_text:
        return make_response(data, "Джемини на связи. Задавай вопрос.")

    # 2. Команда выхода
    if user_lower in {"хватит", "выход", "стоп", "закончить"}:
        if session["task"] and not session["task"].done():
            session["task"].cancel()
        sessions.pop(session_id, None)
        return make_response(data, "Добро, отключаюсь.", end_session=True)

    # 3. Сброс контекста
    if user_lower in {"сброс", "новый диалог", "забудь"}:
        if session["task"] and not session["task"].done():
            session["task"].cancel()
        sessions.pop(session_id, None)
        return make_response(data, "Контекст очищен. О чем поговорим?")

    # 4. Обработка команды «Дальше»
    if user_lower in CONTINUE_WORDS:
        if session["chunks"]:
            return pop_next_chunk_response(data, session)

        task = session["task"]
        if task is not None:
            if not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=YANDEX_TIMEOUT)
                except asyncio.TimeoutError:
                    return make_response(
                        data,
                        "Все еще думаю, вопрос плотный. Скажи «Дальше» через пару секунд.",
                        show_continue_btn=True
                    )

            session["task"] = None
            try:
                full_reply = task.result()
                session["chunks"] = split_into_chunks(full_reply)
                return pop_next_chunk_response(data, session)
            except Exception as e:
                return make_response(data, f"Вылезла ошибка при генерации: {str(e)[:60]}")

        return make_response(data, "В очереди ничего нет. Задавай новый вопрос!")

    # 5. Обработка нового вопроса
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
        return make_response(
            data,
            "Запрос принял, нужно подумать пару секунд. Скажи «Дальше».",
            show_continue_btn=True
        )
    except Exception as e:
        session["task"] = None
        return make_response(data, f"Ошибка связи с Gemini: {str(e)[:60]}")