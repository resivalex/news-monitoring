from typing import List, Optional, TypedDict, Any
import numpy as np


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
        Инициализация DataLoader с путем к CSV-файлу.
        """
        self.file_path = file_path

    def load_next_message(self) -> Optional[Message]:
        """
        Загружает и возвращает следующее сообщение из CSV-файла.
        Возвращает None, если сообщений больше нет.
        """
        pass


# Модуль классификации
class Classifier:
    def __init__(self, term_manager: 'TermManager') -> None:
        """
        Инициализация классификатора с менеджером терминов.
        """
        self.term_manager = term_manager

    def classify(self, text: str) -> bool:
        """
        Классифицирует сообщение как релевантное или нерелевантное.
        Возвращает True, если сообщение связано с Минцифры РФ, иначе False.
        """
        return self.term_manager.contains_relevant_terms(text)


# Модуль управления состоянием
class StateManager:
    def save_embedding(self, embedding: Embedding) -> None:
        """
        Сохраняет эмбеддинг сообщения в памяти.
        """
        pass

    def get_previous_embeddings(self) -> List[Embedding]:
        """
        Возвращает список предыдущих эмбеддингов для сравнения.
        """
        pass


# Модуль уведомлений
class Notifier:
    def notify(self, message: Message) -> None:
        """
        Отправляет уведомление о новом уникальном событии.
        """
        pass


# Модуль поиска терминов
class TermManager:
    def __init__(self, synonyms: List[str]) -> None:
        """
        Инициализация заданным списком синонимов.
        """
        self.synonyms = synonyms

    def contains_relevant_terms(self, text: str) -> bool:
        """
        Проверяет, содержит ли текст релевантные термины (слова и их синонимы).
        Возвращает True, если релевантные термины найдены, иначе False.
        """
        # Приведение текста к нижнему регистру для облегчения поиска
        lower_text = text.lower()

        # Проверка наличия любого синонима в тексте
        for synonym in self.synonyms:
            if synonym.lower() in lower_text:
                return True

        return False

# Модуль предобработки текста
class TextPreprocessor:
    def process_and_embed(self, text: str) -> (str, Embedding):
        """
        Выполняет очистку текста и создание эмбеддинга.
        Возвращает очищенный текст и эмбеддинг.
        """
        pass

# Модуль идентификации уникальных событий
class EventIdentifier:
    def is_event_unique(self, embedding: Embedding) -> bool:
        """
        Проверяет уникальность события на основе эмбеддинга.
        Возвращает True, если событие уникально, иначе False.
        """
        pass

    def save_event_state(self, embedding: Embedding) -> None:
        """
        Сохраняет состояние события (эмбеддинг) для дальнейшего использования.
        """
        pass

# Основной модуль обработки
class MainProcessor:
    def __init__(self, data_loader: DataLoader, text_preprocessor: TextPreprocessor,
                 classifier: Classifier, event_identifier: EventIdentifier,
                 notifier: Notifier) -> None:
        """
        Инициализация MainProcessor с необходимыми модулями.
        """
        self.data_loader = data_loader
        self.text_preprocessor = text_preprocessor
        self.classifier = classifier
        self.event_identifier = event_identifier
        self.notifier = notifier

    def process_next_message(self) -> None:
        """
        Выполняет полную обработку следующего сообщения:
        1. Загружает сообщение.
        2. Предобрабатывает текст и создает эмбеддинг.
        3. Классифицирует релевантность.
        4. Проверяет уникальность события.
        5. Сохраняет состояние события и отправляет уведомление, если событие уникально.
        """
        # Шаг 1: Загрузка следующего сообщения
        message = self.data_loader.load_next_message()
        if message is None:
            print("No more messages to process.")
            return

        # Шаг 2: Предобработка текста и создание эмбеддинга
        cleaned_text, embedding = self.text_preprocessor.process_and_embed(message['text'])

        # Шаг 3: Классификация релевантности сообщения
        if not self.classifier.classify(cleaned_text):
            print(f"Message is not relevant: {message['text']}")
            return

        # Шаг 4: Проверка уникальности события
        if not self.event_identifier.is_event_unique(embedding):
            print(f"Event is not unique, no notification will be sent.")
            return

        # Шаг 5: Сохранение состояния и отправка уведомления
        self.event_identifier.save_event_state(embedding)
        self.notifier.notify(message)
        print(f"Notification sent for message: {message['text']}")
