from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import requests
import logging

import mimetypes
import io
from io import BytesIO
import fitz
from docx import Document
import speech_recognition as sr

import subprocess
import os
import numpy as np

from openai import OpenAI
client = OpenAI(
  api_key="NVIDIA_KEY",
  base_url="https://integrate.api.nvidia.com/v1"
)

FLOWISE_URL = "FLOWISE_URL"

TELEGRAM_TOKEN = "TELEGRAM_TOKEN"

CHUNK_TOKEN_SIZE = 500
APPROX_CHARS_PER_TOKEN = 4
TOP_CHUNKS = 3

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = [
    'text/plain',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
]

def split_text_into_chunks(text, chunk_size=CHUNK_TOKEN_SIZE * APPROX_CHARS_PER_TOKEN):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        if end < text_length:
            sentence_end = end
            while sentence_end > start and text[sentence_end] not in '.!?;\n':
                sentence_end -= 1
            
            if sentence_end > start:
                end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
        start = end
    
    return chunks

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(
                input=[text],
                model="nvidia/nv-embed-v1",
                encoding_format="float",
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Ошибка в get_embeddings: {str(e)}")
            embeddings.append([])
    return embeddings

def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
        
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def get_top_chunks(query, chunks, top_n=TOP_CHUNKS):
    if not chunks:
        return []
    
    texts_to_embed = [query] + chunks
    embeddings = get_embeddings(texts_to_embed)
    
    if not embeddings or not embeddings[0]:
        return chunks[:top_n]
    
    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    
    similarities = []
    for emb in chunk_embeddings:
        if emb:
            similarities.append(cosine_similarity(query_embedding, emb))
        else:
            similarities.append(0.0)
    
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_n]
    
    return [(chunks[i], similarities[i]) for i in top_indices if i < len(chunks)]

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()

    oga_file = BytesIO(file_bytes)
    wav_file = BytesIO()

    process = subprocess.run(['ffmpeg', '-i', '-', '-f', 'wav', '-'], 
                            input=oga_file.getvalue(), 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    wav_file.write(process.stdout)
    wav_file.seek(0)

    r = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="ru-RU")
            await process_flowise_request(update, text)
        except sr.UnknownValueError:
            await update.message.reply_text("Не удалось распознать речь.")
        except sr.RequestError as e:
            await update.message.reply_text(f"Ошибка при распознавании речи: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Это чат бот с рассуждающей моделью, инструменты: google search, calculator, date. Также можно загрузить файлы TXT, PDF, DOC/DOCX и отправить голосовое сообщение")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await process_flowise_request(update, user_message)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    mime_type = document.mime_type or mimetypes.guess_type(document.file_name)[0] or ""
    
    if mime_type not in SUPPORTED_MIME_TYPES:
        await update.message.reply_text("Поддерживаются только файлы: TXT, PDF, DOC/DOCX")
        return
        
    try:
        file = await context.bot.get_file(document.file_id)
        file_bytes = await file.download_as_bytearray()
        
        if mime_type == 'text/plain':
            file_content = file_bytes.decode('utf-8')
        
        elif mime_type == 'application/pdf':
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
                file_content = "\n".join([page.get_text() for page in pdf_doc])

        elif mime_type.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
            doc = Document(io.BytesIO(file_bytes))
            file_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

        else:
            file_content = f"Файл {document.file_name} не обрабатывается"
            return
        
        query = update.message.caption or os.path.splitext(document.file_name)[0]
        
        chunks = split_text_into_chunks(file_content)
        logger.info(f"Разбито на {len(chunks)} чанков")
        
        top_chunks_with_scores = get_top_chunks(query, chunks, TOP_CHUNKS)
        context_parts = []
        for i, (chunk, score) in enumerate(top_chunks_with_scores, 1):
            chunk_header = f"📌 Chunk {i} (Relevance: {score:.2f}):\n"
            context_parts.append(chunk_header + chunk)
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        full_content = f"File: {document.file_name}\n Query: {query}\n\n Most relevant chunks:\n\n{context_text}"
        
        await process_flowise_request(update, full_content)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        await update.message.reply_text("Ошибка при обработке файла.")

async def process_flowise_request(update: Update, question: str):

    chat_id = update.effective_chat.id

    payload = {
        "question": question,
        "overrideConfig": {
        "sessionId": str(chat_id)
        }
    }
    
    try:
        response = requests.post(FLOWISE_URL, json=payload)
        if response.status_code == 200:
            flowise_response = response.json().get("text", "Не получилось обработать ответ.")
            while(len(flowise_response) > 4096):
                temp = flowise_response[:4096]
                await update.message.reply_text(temp)
                flowise_response = flowise_response[4096:]
            await update.message.reply_text(flowise_response)
        else:
            error_msg = f"Flowise error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            await update.message.reply_text("Ошибка при обработке запроса Flowise.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        await update.message.reply_text(f"Произошла критическая ошибка при обработке запроса: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error: {context.error}")
    await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    app.add_error_handler(error_handler)

    logger.info("Бот запущен в режиме polling...")
    app.run_polling()


