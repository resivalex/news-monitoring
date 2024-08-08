from openpyxl import Workbook
from typing import List, Optional, TypedDict, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


# TypedDict for representing the message structure
class NewsMessage(TypedDict):
    timestamp: str
    text: str
    url: str


class Embedding(TypedDict):
    vector: np.ndarray


class ClusterManager:
    def __init__(self) -> None:
        self.embeddings = []
        self.cluster_counter = 1
        self.cluster_ids = []

    def assign_cluster(self, embedding: Embedding) -> int:
        # Save the embedding and assign a new cluster ID
        self.embeddings.append(embedding)
        cluster_id = self.cluster_counter
        self.cluster_ids.append(cluster_id)
        self.cluster_counter += 1
        return cluster_id

    def find_cluster(self, embedding: Embedding, threshold: float = 0.9) -> (bool, Optional[int]):
        # Check if the embedding is unique based on cosine similarity
        for i, stored_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(embedding['vector'], stored_embedding['vector'])
            if similarity >= threshold:
                return False, self.cluster_ids[i]
        return True, None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Calculate cosine similarity between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)


class RelevanceClassifier:
    def __init__(self, keywords: List[str]) -> None:
        self.keywords = keywords

    def is_relevant(self, text: str) -> bool:
        # Classify message as relevant if it contains any of the keywords
        lower_text = text.lower()
        for keyword in self.keywords:
            if keyword.lower() in lower_text:
                return True
        return False


class ExcelWriter:
    def __init__(self, output_file: str, save_interval: int = 1000):
        self.output_file = output_file
        self.workbook = Workbook()
        self.worksheet = self.workbook.active
        self.save_interval = save_interval
        self.row_count = 0

    def write_header(self, columns: List[str]):
        self.worksheet.append(columns)
        self.row_count += 1

    def append_row(self, row: List[Any]):
        # Append a row to the Excel file and save periodically
        self.worksheet.append(row)
        self.row_count += 1

        if self.row_count % self.save_interval == 0:
            self.save()
            print(f"Auto-saved after {self.row_count} rows.")

    def save(self):
        # Save the Excel file
        self.workbook.save(self.output_file)
        print(f"File saved to {self.output_file}")


class TextProcessor:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def preprocess(self, text: str) -> str:
        # Clean and preprocess the text
        text = text.lower()
        text = " ".join(text.split())
        return text

    def generate_embedding(self, text: str) -> Embedding:
        # Generate an embedding for the text using a pre-trained model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return Embedding(vector=embeddings[0])

    def process_and_embed(self, text: str) -> (str, Embedding):
        cleaned_text = self.preprocess(text)
        embedding = self.generate_embedding(cleaned_text)
        return cleaned_text, embedding


class NotificationService:
    def notify(self, message: NewsMessage) -> None:
        # Print notification for new unique event
        print("-------- Notification --------")
        print(f"Time: {message['timestamp']}")
        print(f"URL: {message['url']}")
        print(message["text"])
        print("------------------------------")


class NewsDataLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_and_sort_data()

    def _load_and_sort_data(self) -> List[NewsMessage]:
        # Load data from Excel and sort by publication time
        df = pd.read_excel(self.file_path)
        df['Время публикации'] = pd.to_datetime(df['Время публикации'], errors='coerce')
        df = df.dropna(subset=['Время публикации', 'Текст сообщения'])
        df = df.sort_values(by='Время публикации')
        messages = df.to_dict('records')

        sorted_messages = [
            NewsMessage(
                timestamp=record['Время публикации'].isoformat(),
                text=record['Текст сообщения'],
                url=record['Ссылка на сообщение']
            )
            for record in messages
        ]
        return sorted_messages

    def message_generator(self):
        # Generator for iterating through the loaded messages
        for message in self.data:
            yield message


class NewsProcessor:
    def __init__(self, data_loader: NewsDataLoader, text_processor: TextProcessor,
                 relevance_classifier: RelevanceClassifier, cluster_manager: ClusterManager,
                 notifier: NotificationService, output_file: str) -> None:
        self.data_loader = data_loader
        self.text_processor = text_processor
        self.relevance_classifier = relevance_classifier
        self.cluster_manager = cluster_manager
        self.notifier = notifier
        self.excel_writer = ExcelWriter(output_file)
        self.excel_writer.write_header(['Время публикации', 'Текст сообщения', 'Ссылка на сообщение', 'Кластер', 'Минцифры?'])

    def process_messages(self) -> None:
        for message in self.data_loader.message_generator():
            cleaned_text, embedding = self.text_processor.process_and_embed(message['text'])

            # Classify relevance
            is_relevant = self.relevance_classifier.is_relevant(cleaned_text)
            relevance_flag = 1 if is_relevant else 0

            # Check for uniqueness or assign new cluster
            if not is_relevant:
                cluster_id = self.cluster_manager.assign_cluster(embedding)
            else:
                is_unique, cluster_id = self.cluster_manager.find_cluster(embedding)
                if is_unique:
                    cluster_id = self.cluster_manager.assign_cluster(embedding)
                    self.notifier.notify(message)

            # Write data to Excel
            self.excel_writer.append_row([message['timestamp'], message['text'], message['url'], cluster_id, relevance_flag])

        # Final save of the results
        self.excel_writer.save()

    def calculate_significance(self):
        # Calculate cluster significance by reading the Excel file and updating it
        df = pd.read_excel(self.excel_writer.output_file)

        cluster_counts = df['Кластер'].value_counts()
        df['Значимость'] = df['Кластер'].map(cluster_counts)
        df = df.sort_values(by=['Значимость', 'Кластер', 'Время публикации'], ascending=[False, True, True])

        df.to_excel(self.excel_writer.output_file, index=False)
        print(f"Updated file with 'Значимость' column saved to {self.excel_writer.output_file}")


if __name__ == "__main__":
    file_path = "./data/posts_mc.xlsx"
    output_file = "./data/clustered.xlsx"

    keywords = [
        "минцифры", "минцифра", "минцифре", "минцифрой", "минцифрах",
        "министерство цифрового развития", "министерству цифрового развития",
        "министерства цифрового развития", "министерством цифрового развития",
        "министерство цифровизации", "министерству цифровизации",
        "министерства цифровизации", "министерством цифровизации",
        "цифровое министерство", "цифрового министерства",
        "цифровому министерству", "цифровым министерством",
        "министерство цифровой экономики", "министерству цифровой экономики",
        "министерства цифровой экономики", "министерством цифровой экономики",
        "министерство цифровых технологий", "министерству цифровых технологий",
        "министерства цифровых технологий", "министерством цифровых технологий"
    ]

    data_loader = NewsDataLoader(file_path)
    relevance_classifier = RelevanceClassifier(keywords)
    cluster_manager = ClusterManager()
    text_processor = TextProcessor()
    notifier = NotificationService()

    news_processor = NewsProcessor(
        data_loader=data_loader,
        text_processor=text_processor,
        relevance_classifier=relevance_classifier,
        cluster_manager=cluster_manager,
        notifier=notifier,
        output_file=output_file
    )

    # Process all messages
    news_processor.process_messages()

    # Calculate and add the "Significance" column
    news_processor.calculate_significance()
