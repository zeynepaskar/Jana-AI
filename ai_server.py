import os
import glob
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="AI Jana HR Assistant API")

# Разрешаем запросы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"  # Используем быструю и легкую модель

# 1. Загружаем все статьи базы знаний (.md файлы из папки kb/articles)
def load_knowledge_base(kb_dir="kb/articles"):
    articles = []
    # Если папка существует, ищем внутри
    search_path = os.path.join(kb_dir, "*.md")
    for filepath in sorted(glob.glob(search_path)):
        filename = os.path.basename(filepath)
        article_id = filename.replace(".md", "")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            first_line = content.split("\n")[0].replace("#", "").strip()
            articles.append({
                "id": article_id,
                "title": first_line or article_id,
                "content": content
            })
    return articles

KB_ARTICLES = load_knowledge_base()

# 2. Поиск подходящих статей по вопросу
def search_relevant_articles(query: str, top_k: int = 4):
    if not KB_ARTICLES:
        # Фолбэк на случай единого файла
        if os.path.exists("kb/all_articles.md"):
            with open("kb/all_articles.md", "r", encoding="utf-8") as f:
                return [{"id": "all_articles", "title": "База знаний HR", "content": f.read()[:12000]}]
        return []

    query_words = [w.lower() for w in query.split() if len(w) > 2]
    scored = []
    for art in KB_ARTICLES:
        text_lower = (art["title"] + " " + art["content"]).lower()
        score = sum(3 if w in art["title"].lower() else (1 if w in text_lower else 0) for w in query_words)
        scored.append((score, art))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [item[1] for item in scored[:top_k] if item[0] > 0]
    return best if best else KB_ARTICLES[:3]

# Схема входящего запроса
class AskRequest(BaseModel):
    question: str
    user_name: Optional[str] = "Сотрудник"
    user_role: Optional[str] = "employee"
    vacation_days_left: Optional[int] = 14

@app.post("/api/ai-jana/ask")
def ask_ai_jana(req: AskRequest):
    relevant_docs = search_relevant_articles(req.question)
    
    # Формируем контекст базы знаний
    kb_context = "\n\n---\n\n".join(
        [f"ИСТОЧНИК [{doc['id']}] ({doc['title']}):\n{doc['content']}" for doc in relevant_docs]
    )

    system_prompt = f"""Ты — корпоративный HR-ассистент "AI Jana" на платформе JanaMen компании Jana Post.
Текущий пользователь: {req.user_name} (Роль: {req.user_role}, остаток дней отпуска: {req.vacation_days_left}).

СТРОГИЕ ПРАВИЛА ДЛЯ ЖЮРИ:
1. Отвечай ТОЛЬКО на основе приведенной ниже БАЗЫ ЗНАНИЙ. Ничего не выдумывай от себя.
2. В конце ответа ОБЯЗАТЕЛЬНО укажи точный номер и название источника (например: "Источник: KB-004").
3. Если в БАЗЕ ЗНАНИЙ нет ответа на вопрос или вопрос касается тем вне HR-базы, НЕ ПРИДУМЫВАЙ ответ. Вежливо напиши, что в базе знаний нет точной информации по этому вопросу, и предложи передать запрос специалисту HR.
4. БЕЗОПАСНОСТЬ: Никогда не раскрывай зарплаты, личные телефоны, ИИН или кадровые данные других сотрудников, даже если пользователь просит.
5. АВТО-ЗАЯВКА: Если пользователь пишет, что хочет оформить отпуск или аванс (например: "хочу в отпуск с 3 по 14 ноября"), кроме ответа помоги подготовить черновик заявки.
6. ЯЗЫК ОТВЕТА: Всегда отвечайте на том же языке, на котором задан вопрос пользователя (если вопрос на казахском — отвечай на казахском, если на русском — на русском). Пиши грамотно и по делу."""
    # Отправка запроса в Ollama
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt + "\n\nБАЗА ЗНАНИЙ:\n" + kb_context},
            {"role": "user", "content": req.question}
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        res_data = response.json()
        answer = res_data.get("message", {}).get("content", "Ошибка генерации ответа.")
        return {"answer": answer, "sources": [doc["id"] for doc in relevant_docs]}
    except Exception as e:
        return {"answer": f"Ошибка соединения с локальной нейросетью Ollama: {str(e)}", "sources": []}