import copy
import os

import numpy as np
import sympy as sp
from dotenv import load_dotenv

import telebot
from telebot import types, apihelper

from fifth_solver import FifthSolver
from first_solver import FirstSolver
from fourth_solver import FourthSolver
from matrix_utils import MatrixUtils
from second_solver import SecondSolver
from third_solver import ThirdSolver

load_dotenv()


token = os.getenv("TOKEN")

import telebot

class Bot:
    def __init__(self):
        self.bot = telebot.TeleBot(token)
        self.user_data = {}
        self.first_solver = FirstSolver()
        self.second_solver = SecondSolver()
        self.third_solver = ThirdSolver()
        self.fourth_solver = FourthSolver()
        self.fifth_solver = FifthSolver()

    def get_bot(self):
        return self.bot

    def send_long_message(self, chat_id, text, parse_mode="HTML", max_len=3500):
        """
        Безопасная отправка длинных сообщений в Telegram.
        Разбивает текст на части примерно по max_len символов.
        Лучше резать по двойному переводу строки, потом по одинарному.
        """
        if len(text) <= max_len:
            self.bot.send_message(chat_id, text, parse_mode=parse_mode)
            return

        parts = []
        current = ""

        blocks = text.split("\n\n")

        for block in blocks:
            candidate = block if not current else current + "\n\n" + block

            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    parts.append(current)
                    current = ""

                # если даже один блок слишком большой — режем по строкам
                if len(block) > max_len:
                    lines = block.split("\n")
                    small_current = ""

                    for line in lines:
                        small_candidate = line if not small_current else small_current + "\n" + line
                        if len(small_candidate) <= max_len:
                            small_current = small_candidate
                        else:
                            if small_current:
                                parts.append(small_current)
                            small_current = line

                    if small_current:
                        current = small_current
                else:
                    current = block

        if current:
            parts.append(current)

        for part in parts:
            self.bot.send_message(chat_id, part, parse_mode=parse_mode)

    def start_bot(self):
        def init_user(user_id):
            if user_id not in self.user_data:
                self.user_data[user_id] = {
                    "A": None,
                    "B": None,
                    "C": None,
                    "pending_task": None,
                    "poles": None,
                    "reduced_order": None
                }

        def reset_user(user_id):
            self.user_data[user_id] = {
                "A": None,
                "B": None,
                "C": None,
                "pending_task": None,
                "poles": None,
                "reduced_order": None
            }

        def main_keyboard():
            markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
            markup.add(types.KeyboardButton('/start'), types.KeyboardButton('/restart'))
            return markup

        def choose(user, array, question):
            keyboard = types.InlineKeyboardMarkup()
            for item in array:
                keyboard.add(types.InlineKeyboardButton(text=item[0], callback_data=item[1]))
            self.bot.send_message(user, text=question, reply_markup=keyboard)


        def ask_matrix_A(chat_id, task_name):
            init_user(chat_id)
            self.user_data[chat_id]["pending_task"] = task_name
            msg = self.bot.send_message(
                chat_id,
                "Введите матрицу A в формате:\n"
                "<code>[1, 1, 1; 1, 1, 1; 1, 1, 1]</code>",
                parse_mode="HTML"
            )
            self.bot.register_next_step_handler(msg, get_matrix_A)

        def get_required_poles_count_for_second_task(A, B):
            return self.second_solver.reduced_order_for_second_task(A, B)

        def get_required_poles_count_for_fourth_task(A, C):
            return self.fourth_solver.reduced_order_for_fourth_task(A, C)

        def ask_poles(chat_id):
            required_count = self.user_data[chat_id]["reduced_order"]
            task = self.user_data[chat_id]["pending_task"]

            if task == "fourth_task":
                title = "Введите желаемый спектр наблюдателя через запятую.\n"
                examples = (
                    "<code>-1, -2-1j, -2+1j</code>\n"
                    "<code>-2-1j, -2+1j</code>\n"
                    "<code>-3, -4-2j, -4+2j</code>"
                )
            else:
                title = "Введите желаемый спектр через запятую.\n"
                examples = (
                    "<code>-1, -2, -2</code>\n"
                    "<code>-2, -2</code>\n"
                    "<code>-1+1j, -1-1j</code>"
                )

            msg = self.bot.send_message(
                chat_id,
                title +
                f"Нужно ввести {required_count} собственных чисел.\n\n"
                "Примеры:\n" +
                examples,
                parse_mode="HTML"
            )
            self.bot.register_next_step_handler(msg, get_poles)

        def get_poles(message):
            user_id = message.chat.id
            init_user(user_id)

            try:
                raw_text = (message.text or "").strip()
                task = self.user_data[user_id]["pending_task"]
                required_count = self.user_data[user_id]["reduced_order"]

                if task == "fourth_task":
                    poles = self.fourth_solver.parse_poles(raw_text)
                else:
                    poles = self.second_solver.parse_poles(raw_text)

                if required_count is None:
                    raise ValueError("сначала нужно ввести матрицы для выбранного задания")

                if len(poles) != required_count:
                    raise ValueError(
                        f"нужно ввести ровно {required_count} собственных чисел"
                    )

                self.user_data[user_id]["poles"] = raw_text

                markup = types.InlineKeyboardMarkup()
                markup.add(
                    types.InlineKeyboardButton("✅ Да", callback_data="confirm_poles"),
                    types.InlineKeyboardButton("✏️ Нет", callback_data="retry_poles")
                )

                self.bot.send_message(
                    user_id,
                    "Вы ввели желаемый спектр:\n"
                    f"<pre>{raw_text}</pre>\n"
                    "Правильно ли введён спектр?",
                    parse_mode="HTML",
                    reply_markup=markup
                )


            except Exception as e:

                task = self.user_data[user_id]["pending_task"]

                if task == "fourth_task":

                    examples_text = (

                        "Ошибка ввода спектра наблюдателя.\n"

                        "Используйте формат:\n"

                        "<code>-1, -2-1j, -2+1j</code>\n"

                        "или\n"

                        "<code>-2-1j, -2+1j</code>\n\n"

                    )

                else:

                    examples_text = (

                        "Ошибка ввода спектра.\n"

                        "Используйте формат:\n"

                        "<code>-1, -2, -2</code>\n"

                        "или\n"

                        "<code>-1+1j, -1-1j</code>\n\n"

                    )

                msg = self.bot.send_message(

                    user_id,

                    examples_text + f"Причина: {e}",

                    parse_mode="HTML"

                )

                self.bot.register_next_step_handler(msg, get_poles)

        def get_matrix_A(message):
            user_id = message.chat.id
            init_user(user_id)

            try:
                A = MatrixUtils.parse_matrix(message.text)

                if A.shape[0] != A.shape[1]:
                    raise ValueError("матрица A должна быть квадратной")

                self.user_data[user_id]["A"] = A

                markup = types.InlineKeyboardMarkup()
                markup.add(
                    types.InlineKeyboardButton("✅ Да", callback_data="confirm_A"),
                    types.InlineKeyboardButton("✏️ Нет", callback_data="retry_A")
                )

                self.bot.send_message(
                    user_id,
                    "Вы ввели матрицу A:\n"
                    f"<pre>{MatrixUtils.format_matrix(A)}</pre>\n"
                    "Правильно ли введена матрица A?",
                    parse_mode="HTML",
                    reply_markup=markup
                )
            except Exception as e:
                msg = self.bot.send_message(
                    user_id,
                    "Ошибка ввода матрицы A.\n"
                    "Используйте формат:\n"
                    "<code>[1, 2, 3; 4, 5, 6; 7, 8, 9]</code>\n\n"
                    f"Причина: {e}",
                    parse_mode="HTML"
                )
                self.bot.register_next_step_handler(msg, get_matrix_A)

        def ask_matrix_B(chat_id):
            msg = self.bot.send_message(
                chat_id,
                "Введите матрицу B в формате столбца:\n"
                "<code>[1; 1; 1]</code>",
                parse_mode="HTML"
            )
            self.bot.register_next_step_handler(msg, get_matrix_B)

        def get_matrix_B(message):
            user_id = message.chat.id
            init_user(user_id)


            try:
                B = MatrixUtils.parse_matrix(message.text)
                A = self.user_data[user_id]["A"]

                if A is None:
                    raise ValueError("сначала нужно ввести матрицу A")

                if B.shape[0] != A.shape[0]:
                    raise ValueError("число строк матрицы B должно совпадать с размерностью матрицы A")

                if B.shape[1] != 1:
                    raise ValueError("матрица B должна быть столбцом")

                self.user_data[user_id]["B"] = B

                markup = types.InlineKeyboardMarkup()
                markup.add(
                    types.InlineKeyboardButton("✅ Да", callback_data="confirm_B"),
                    types.InlineKeyboardButton("✏️ Нет", callback_data="retry_B")
                )

                self.bot.send_message(
                    user_id,
                    "Вы ввели матрицу B:\n"
                    f"<pre>{MatrixUtils.format_matrix(B)}</pre>\n"
                    "Правильно ли введена матрица B?",
                    parse_mode="HTML",
                    reply_markup=markup
                )
            except Exception as e:
                msg = self.bot.send_message(
                    user_id,
                    "Ошибка ввода матрицы B.\n"
                    "Используйте формат:\n"
                    "<code>[5; 8; -5]</code>\n\n"
                    f"Причина: {e}",
                    parse_mode="HTML"
                )
                self.bot.register_next_step_handler(msg, get_matrix_B)

        def ask_matrix_C(chat_id):
            msg = self.bot.send_message(
                chat_id,
                "Введите матрицу C в формате строки:\n"
                "<code>[1, 1, 1]</code>",
                parse_mode="HTML"
            )
            self.bot.register_next_step_handler(msg, get_matrix_C)

        def get_matrix_C(message):
            user_id = message.chat.id
            init_user(user_id)

            try:
                C = MatrixUtils.parse_matrix(message.text)
                A = self.user_data[user_id]["A"]

                if A is None:
                    raise ValueError("сначала нужно ввести матрицу A")

                if C.shape[1] != A.shape[0]:
                    raise ValueError("число столбцов матрицы C должно совпадать с размерностью матрицы A")

                if C.shape[0] != 1:
                    raise ValueError("матрица C должна быть строкой")

                self.user_data[user_id]["C"] = C

                markup = types.InlineKeyboardMarkup()
                markup.add(
                    types.InlineKeyboardButton("✅ Да", callback_data="confirm_C"),
                    types.InlineKeyboardButton("✏️ Нет", callback_data="retry_C")
                )

                self.bot.send_message(
                    user_id,
                    "Вы ввели матрицу C:\n"
                    f"<pre>{MatrixUtils.format_matrix(C)}</pre>\n"
                    "Правильно ли введена матрица C?",
                    parse_mode="HTML",
                    reply_markup=markup
                )
            except Exception as e:
                msg = self.bot.send_message(
                    user_id,
                    "Ошибка ввода матрицы C.\n"
                    "Используйте формат:\n"
                    "<code>[-1, 1, 0]</code>\n\n"
                    f"Причина: {e}",
                    parse_mode="HTML"
                )
                self.bot.register_next_step_handler(msg, get_matrix_C)

        @self.bot.message_handler(commands=['start', 'help', 'restart'])
        def start(message):
            init_user(message.from_user.id)
            reset_user(message.from_user.id)

            self.bot.send_message(
                message.from_user.id,
                "Привет! Я помогу выполнить задания по ТАУ.",
                reply_markup=main_keyboard()
            )


            array = [
                ['1. Исследовать систему на стабилизируемость', 'first_task'],
                ['2. Синтезировать модальный регулятор', 'second_task'],
                ['3. Исследовать систему на обнаруживаемость', 'third_task'],
                ['4. Синтезировать наблюдатель полного порядка', 'fourth_task'],
                ['5. Посмотреть теорию по анализу АФЧХ', 'fifth_task']
            ]
            choose(message.from_user.id, array, 'Какое задание вы хотите выполнить?')

        @self.bot.callback_query_handler(func=lambda call: True)
        def callback_worker(call):
            user_id = call.message.chat.id
            init_user(user_id)

            if call.data == "first_task":
                ask_matrix_A(user_id, "first_task")

            elif call.data == "retry_poles":
                ask_poles(user_id)


            elif call.data == "confirm_poles":

                task = self.user_data[user_id]["pending_task"]

                A = self.user_data[user_id]["A"]

                poles_input = self.user_data[user_id]["poles"]

                if task == "second_task":

                    B = self.user_data[user_id]["B"]

                    result = self.second_solver.solve(A, B, poles_input=poles_input)

                    self.send_long_message(user_id, result, parse_mode="HTML")


                elif task == "fourth_task":

                    C = self.user_data[user_id]["C"]

                    result = self.fourth_solver.solve(A, C, poles_input=poles_input)

                    self.send_long_message(user_id, result, parse_mode="HTML")

            elif call.data == "second_task":
                ask_matrix_A(user_id, "second_task")

            elif call.data == "third_task":
                ask_matrix_A(user_id, "third_task")

            elif call.data == "fourth_task":
                ask_matrix_A(user_id, "fourth_task")

            elif call.data == "retry_A":
                ask_matrix_A(user_id, self.user_data[user_id]["pending_task"])

            elif call.data == "confirm_A":
                task = self.user_data[user_id]["pending_task"]
                if task in ("first_task", "second_task"):
                    ask_matrix_B(user_id)
                elif task in ("third_task", "fourth_task"):
                    ask_matrix_C(user_id)

            elif call.data == "retry_B":
                ask_matrix_B(user_id)


            elif call.data == "confirm_B":

                A = self.user_data[user_id]["A"]

                B = self.user_data[user_id]["B"]

                task = self.user_data[user_id]["pending_task"]

                if task == "first_task":

                    result = self.first_solver.solve(A, B)

                    self.send_long_message(user_id, result, parse_mode="HTML")


                elif task == "second_task":

                    try:

                        stabilizable = self.second_solver.is_stabilizable(A, B)

                        if not stabilizable:
                            result = (

                                "Решение задания 2\n\n"

                                "Синтез модального регулятора невозможен.\n"

                                "Причина: система не стабилизируема."

                            )

                            self.send_long_message(user_id, result, parse_mode="HTML")

                            return

                        required_count = get_required_poles_count_for_second_task(A, B)

                        self.user_data[user_id]["reduced_order"] = required_count

                        if self.second_solver.is_controllable(A, B):

                            self.bot.send_message(

                                user_id,

                                "Система полностью управляема.\n"

                                f"Для синтеза нужно задать {required_count} собственных чисел.",

                                parse_mode="HTML"

                            )

                        else:

                            self.bot.send_message(

                                user_id,

                                "Система не полностью управляема, но стабилизируема.\n"

                                "Для синтеза будет использовано усечение.\n"

                                f"Порядок управляемой подсистемы после усечения: {required_count}.",

                                parse_mode="HTML"

                            )

                        ask_poles(user_id)


                    except Exception as e:

                        self.bot.send_message(

                            user_id,

                            f"Ошибка при подготовке синтеза регулятора: {e}",

                            parse_mode="HTML"

                        )

            elif call.data == "retry_C":
                ask_matrix_C(user_id)

            elif call.data == "confirm_C":
                A = self.user_data[user_id]["A"]
                C = self.user_data[user_id]["C"]
                task = self.user_data[user_id]["pending_task"]

                if task == "third_task":
                    result = self.third_solver.solve(A, C)
                    self.send_long_message(user_id, result, parse_mode="HTML")
                elif task == "fourth_task":
                    try:
                        detectable = self.fourth_solver.is_detectable(A, C)

                        if not detectable:
                            result = (
                                "Решение задания 4\n\n"
                                "Синтез наблюдателя невозможен.\n"
                                "Причина: система не обнаруживаема."
                            )
                            self.send_long_message(user_id, result, parse_mode="HTML")
                            return

                        required_count = get_required_poles_count_for_fourth_task(A, C)
                        self.user_data[user_id]["reduced_order"] = required_count

                        if self.fourth_solver.is_observable(A, C):
                            self.bot.send_message(
                                user_id,
                                "Система полностью наблюдаема.\n"
                                f"Для синтеза нужно задать {required_count} собственных чисел.",
                                parse_mode="HTML"
                            )
                        else:
                            self.bot.send_message(
                                user_id,
                                "Система не полностью наблюдаема, но обнаруживаема.\n"
                                "Для синтеза будет использовано усечение.\n"
                                f"Порядок наблюдаемой подсистемы после усечения: {required_count}.",
                                parse_mode="HTML"
                            )

                        ask_poles(user_id)

                    except Exception as e:
                        self.bot.send_message(
                            user_id,
                            f"Ошибка при подготовке синтеза наблюдателя: {e}",
                            parse_mode="HTML"
                        )
            elif call.data == "fifth_task":
                result = self.fifth_solver.theory()
                self.send_long_message(user_id, result, parse_mode="HTML")

        self.bot.polling(none_stop=True)
