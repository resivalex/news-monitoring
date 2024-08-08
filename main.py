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


class EventStateManager:
    def __init__(self) -> None:
        """
        Инициализация EventStateManager.
        Создает пустой список для хранения эмбеддингов сообщений и их кластеров.
        """
        self.embeddings = []
        self.cluster_counter = 1  # Счетчик кластеров
        self.clusters = []  # Список кластеров, соответствующих эмбеддингам

    def save_event_state(self, embedding: Embedding) -> int:
        """
        Сохраняет эмбеддинг сообщения в память и присваивает ему номер кластера.
        Возвращает номер кластера.
        """
        self.embeddings.append(embedding)
        cluster_id = self.cluster_counter
        self.clusters.append(cluster_id)
        self.cluster_counter += 1
        return cluster_id

    def is_event_unique(self, embedding: Embedding, threshold: float = 0.9) -> (bool, Optional[int]):
        """
        Проверяет уникальность события на основе эмбеддинга.
        Сравнение осуществляется с использованием косинусного расстояния.
        Возвращает кортеж (True, None) если событие уникально,
        либо (False, номер кластера) если это дубликат.
        """
        for i, stored_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(embedding['vector'], stored_embedding['vector'])
            if similarity >= threshold:
                return False, self.clusters[i]
        return True, None

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


# Основной модуль обработки
class MainProcessor:
    def __init__(self, data_loader: DataLoader, text_preprocessor: TextPreprocessor,
                 text_classifier: TextClassifier, event_state_manager: EventStateManager,
                 notifier: Notifier) -> None:
        """
        Инициализация MainProcessor с необходимыми модулями.
        """
        self.data_loader = data_loader
        self.text_preprocessor = text_preprocessor
        self.text_classifier = text_classifier
        self.event_state_manager = event_state_manager
        self.notifier = notifier
        self.results = []  # Список для хранения результатов

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
                self.results.append({**message, "Кластер": 0})
                continue

            # Шаг 3: Проверка уникальности события
            is_unique, cluster_id = self.event_state_manager.is_event_unique(embedding)
            if not is_unique:
                duplicate_events += 1
                self.results.append({**message, "Кластер": cluster_id})
                continue

            # Шаг 4: Сохранение состояния и присвоение нового кластера
            new_cluster_id = self.event_state_manager.save_event_state(embedding)
            self.results.append({**message, "Кластер": new_cluster_id})

            # Отправка уведомления
            self.notifier.notify(message)

        # Сохранение результатов в файл
        self.save_results_to_excel()

    def save_results_to_excel(self) -> None:
        """
        Сохраняет результаты обработки в XLSX файл.
        """
        df = pd.DataFrame(self.results)
        output_file = "./data/clustered.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")


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
    event_state_manager = EventStateManager()
    text_preprocessor = TextPreprocessor()
    notifier = Notifier()

    # Создание основного процессора
    main_processor = MainProcessor(
        data_loader=data_loader,
        text_preprocessor=text_preprocessor,
        text_classifier=text_classifier,
        event_state_manager=event_state_manager,
        notifier=notifier
    )

    # Запуск обработки всех сообщений
    main_processor.process_messages()
