import os
from dotenv import load_dotenv
import telebot

load_dotenv()

TOKEN = (
    os.getenv("TOKEN")
    or os.getenv("BOT_TOKEN")
    or os.getenv("BOT_API_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
)

WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "tau-kr.bothost.tech")
PORT = int(os.getenv("PORT", "3000"))

if not TOKEN:
    raise ValueError("Не найден токен бота")

WEBHOOK_PATH = f"/webhook/{TOKEN}"
WEBHOOK_URL = f"https://{WEBHOOK_HOST}{WEBHOOK_PATH}"

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=["start"])
def start_handler(message):
    print(f"/start from chat_id={message.chat.id}", flush=True)
    bot.send_message(message.chat.id, "Привет, webhook работает")

@bot.message_handler(content_types=["text"])
def echo_handler(message):
    print(
        f"text update chat_id={message.chat.id} text={message.text!r}",
        flush=True
    )
    bot.send_message(message.chat.id, f"Ты написал: {message.text}")

if __name__ == "__main__":
    print("Starting webhook bot...", flush=True)
    print("PORT =", PORT, flush=True)
    print("WEBHOOK_PATH =", WEBHOOK_PATH, flush=True)
    print("WEBHOOK_URL =", WEBHOOK_URL, flush=True)

    try:
        bot.remove_webhook()
        print("remove_webhook ok", flush=True)
    except Exception as e:
        print(f"remove_webhook error: {e}", flush=True)

    bot.run_webhooks(
        listen="0.0.0.0",
        port=PORT,
        url_path=WEBHOOK_PATH,
        webhook_url=WEBHOOK_URL,
        allowed_updates=["message", "callback_query"],
        max_connections=1,
        drop_pending_updates=True,
    )