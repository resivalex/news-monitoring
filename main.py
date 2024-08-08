from typing import List, Optional, TypedDict, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


# TypedDict для представления структуры сообщения
class Message(TypedDict):
    timestamp: str
    text: str
    url: str


class Embedding(TypedDict):
    vector: np.ndarray


# Модуль загрузки данных
class DataLoader:
    def __init__(self, file_path: str) -> None:
        """
        Инициализация DataLoader с путем к XLSX-файлу.
        Загружает данные из файла и сортирует их по времени публикации.
        """
        self.file_path = file_path
        self.data = self._load_and_sort_data()

    def _load_and_sort_data(self) -> List[Message]:
        """
        Внутренний метод для загрузки данных из XLSX-файла и сортировки по времени публикации.
        """
        # Загружаем данные из XLSX-файла
        df = pd.read_excel(self.file_path)

        # Преобразуем колонку "Время публикации" в datetime
        df['Время публикации'] = pd.to_datetime(df['Время публикации'], errors='coerce')

        # Удаляем строки с некорректными датами или пустыми значениями
        df = df.dropna(subset=['Время публикации', 'Текст сообщения'])

        # Сортируем по времени публикации
        df = df.sort_values(by='Время публикации')

        # Преобразуем данные в список словарей
        messages = df.to_dict('records')

        # Преобразуем записи в формат Message
        sorted_messages = [
            Message(
                timestamp=record['Время публикации'].isoformat(),
                text=record['Текст сообщения'],
                url=record['Ссылка на сообщение']
            )
            for record in messages
        ]

        return sorted_messages

    def message_generator(self):
        """
        Генератор, который последовательно возвращает сообщения из загруженных и отсортированных данных.
        """
        for message in self.data:
            yield message


# Объединенный модуль поиска терминов и классификации
class TextClassifier:
    def __init__(self, synonyms: List[str]) -> None:
        """
        Инициализация классификатора с заданным списком синонимов.
        """
        self.synonyms = synonyms

    def classify(self, text: str) -> bool:
        """
        Классифицирует сообщение как релевантное или нерелевантное.
        Возвращает True, если сообщение связано с термином, иначе False.
        """
        lower_text = text.lower()
        for synonym in self.synonyms:
            if synonym.lower() in lower_text:
                return True
        return False


# Модуль управления состоянием
class StateManager:
    def __init__(self) -> None:
        """
        Инициализация StateManager.
        Создает пустой список для хранения эмбеддингов сообщений.
        """
        self.embeddings = []

    def save_embedding(self, embedding: Embedding) -> None:
        """
        Сохраняет эмбеддинг сообщения в память.
        """
        self.embeddings.append(embedding)

    def get_previous_embeddings(self) -> List[Embedding]:
        """
        Возвращает список предыдущих эмбеддингов для сравнения.
        """
        return self.embeddings

    def is_duplicate(self, new_embedding: Embedding, threshold: float = 0.9) -> bool:
        """
        Проверяет, является ли новый эмбеддинг дубликатом одного из ранее сохраненных.
        Сравнение осуществляется с использованием косинусного расстояния.
        Возвращает True, если найден дубликат, иначе False.
        """
        for stored_embedding in self.embeddings:
            similarity = self._cosine_similarity(new_embedding['vector'], stored_embedding['vector'])
            if similarity >= threshold:
                return True
        return False

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)


# Модуль уведомлений
class Notifier:
    def notify(self, message: Message) -> None:
        """
        Отправляет уведомление о новом уникальном событии.
        Выводит информацию о сообщении в консоль.
        """
        print("-------- Notification --------")
        print(f"Time: {message['timestamp']}")
        print(f"URL: {message['url']}")
        print(message["text"])
        print("------------------------------")


# Модуль предобработки текста
class TextPreprocessor:
    def __init__(self) -> None:
        """
        Инициализация TextPreprocessor с загрузкой модели и токенизатора.
        """
        # Загружаем токенизатор и модель
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def preprocess(self, text: str) -> str:
        """
        Выполняет очистку текста: приведение к нижнему регистру, удаление пунктуации и лишних пробелов.
        Возвращает очищенный текст.
        """
        # Приведение текста к нижнему регистру
        text = text.lower()

        # Удаление лишних пробелов
        text = " ".join(text.split())

        return text

    def create_embedding(self, text: str) -> Embedding:
        """
        Создает эмбеддинг текста с использованием предобученной модели.
        Возвращает эмбеддинг в виде numpy массива.
        """
        # Токенизация текста
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Генерация эмбеддингов с использованием модели
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Получаем эмбеддинги
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return Embedding(vector=embeddings[0])

    def process_and_embed(self, text: str) -> (str, Embedding):
        """
        Выполняет очистку текста и создание эмбеддинга.
        Возвращает очищенный текст и эмбеддинг.
        """
        cleaned_text = self.preprocess(text)
        embedding = self.create_embedding(cleaned_text)
        return cleaned_text, embedding


# Модуль идентификации уникальных событий
class EventIdentifier:
    def __init__(self, state_manager: StateManager) -> None:
        """
        Инициализация EventIdentifier с менеджером состояния.
        """
        self.state_manager = state_manager

    def is_event_unique(self, embedding: Embedding) -> bool:
        """
        Проверяет уникальность события на основе эмбеддинга.
        Возвращает True, если событие уникально, иначе False.
        """
        # Проверяем, является ли новый эмбеддинг дубликатом
        if self.state_manager.is_duplicate(embedding):
            return False
        return True

    def save_event_state(self, embedding: Embedding) -> None:
        """
        Сохраняет состояние события (эмбеддинг) для дальнейшего использования.
        """
        self.state_manager.save_embedding(embedding)


# Основной модуль обработки
class MainProcessor:
    def __init__(self, data_loader: DataLoader, text_preprocessor: TextPreprocessor,
                 text_classifier: TextClassifier, event_identifier: EventIdentifier,
                 notifier: Notifier) -> None:
        """
        Инициализация MainProcessor с необходимыми модулями.
        """
        self.data_loader = data_loader
        self.text_preprocessor = text_preprocessor
        self.text_classifier = text_classifier
        self.event_identifier = event_identifier
        self.notifier = notifier

    def process_messages(self) -> None:
        """
        Выполняет полную обработку всех сообщений, используя генератор данных.
        """
        total_messages = 0
        unrelevant_messages = 0
        duplicate_events = 0
        for message in self.data_loader.message_generator():
            if total_messages % 1000 == 0:
                print(f"Total / Unrelevant / Duplicates: {total_messages} / {unrelevant_messages} / {duplicate_events}")
            total_messages += 1
            # Шаг 1: Предобработка текста и создание эмбеддинга
            cleaned_text, embedding = self.text_preprocessor.process_and_embed(message['text'])

            # Шаг 2: Классификация релевантности сообщения
            if not self.text_classifier.classify(cleaned_text):
                unrelevant_messages += 1
                continue

            # Шаг 3: Проверка уникальности события
            if not self.event_identifier.is_event_unique(embedding):
                duplicate_events += 1
                continue

            # Шаг 4: Сохранение состояния и отправка уведомления
            self.event_identifier.save_event_state(embedding)
            self.notifier.notify(message)


if __name__ == "__main__":
    # Пусть к файлу с новостями
    file_path = "./data/posts_mc.xlsx"

    # Определяем список синонимов для фильтрации новостей
    synonyms = [
        "минцифры рф", "минцифра рф", "минцифре рф", "минцифрой рф", "минцифрах рф",
        "министерство цифрового развития", "министерству цифрового развития",
        "министерства цифрового развития", "минцифра россии", "минцифры россии",
        "министерство цифрового развития рф", "министерство цифрового развития россии",
        "министерство цифровизации", "министерство цифровизации рф",
        "министерство цифровизации россии", "минцифры", "минцифра",
        "цифровое министерство", "цифровое министерство рф",
        "цифровое министерство россии", "министерство цифровой экономики",
        "министерство цифровой экономики рф", "министерство цифровой экономики россии",
        "минциф", "минцифров", "минцифрами", "министерство цифрового развития и связи",
        "министерство цифрового развития и связи рф", "министерство цифрового развития и связи россии",
        "министерство цифровых технологий", "министерство цифровых технологий рф",
        "министерство цифровых технологий россии"
    ]

    # Инициализация модулей
    data_loader = DataLoader(file_path)
    text_classifier = TextClassifier(synonyms)
    state_manager = StateManager()
    event_identifier = EventIdentifier(state_manager)
    text_preprocessor = TextPreprocessor()
    notifier = Notifier()

    # Создание основного процессора
    main_processor = MainProcessor(
        data_loader=data_loader,
        text_preprocessor=text_preprocessor,
        text_classifier=text_classifier,
        event_identifier=event_identifier,
        notifier=notifier
    )

    # Запуск обработки всех сообщений
    main_processor.process_messages()
