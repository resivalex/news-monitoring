from openpyxl import Workbook
from typing import List, Optional, TypedDict, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# TypedDict for representing a news entry structure
class NewsEntry(TypedDict):
    timestamp: str
    text: str
    url: str


class TextVector(TypedDict):
    vector: np.ndarray


class ClusterHandler:
    def __init__(self) -> None:
        self.embeddings = []
        self.cluster_count = 1
        self.cluster_ids = []

    def assign_cluster_id(self, embedding: TextVector) -> int:
        # Save the embedding and assign a new cluster ID
        self.embeddings.append(embedding)
        cluster_id = self.cluster_count
        self.cluster_ids.append(cluster_id)
        self.cluster_count += 1
        return cluster_id

    def find_cluster(self, embedding: TextVector, threshold: float = 0.9) -> (bool, Optional[int]):
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


class KeywordMatcher:
    def __init__(self, keywords: List[str]) -> None:
        self.keywords = keywords

    def matches_keywords(self, text: str) -> bool:
        # Check if the text contains any of the relevant keywords
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)


class SpreadsheetWriter:
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


class TextPipeline:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def clean_text(self, text: str) -> str:
        # Clean and preprocess the text
        return " ".join(text.lower().split())

    def vectorize_text(self, text: str) -> TextVector:
        # Generate an embedding for the text using a pre-trained model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return TextVector(vector=embeddings[0])

    def process_text(self, text: str) -> (str, TextVector):
        cleaned_text = self.clean_text(text)
        embedding = self.vectorize_text(cleaned_text)
        return cleaned_text, embedding


class Notifier:
    def notify(self, message: NewsEntry) -> None:
        # Print notification for a new unique event
        print("-------- Notification --------")
        print(f"Time: {message['timestamp']}")
        print(f"URL: {message['url']}")
        print(message["text"])
        print("------------------------------")


class NewsReader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_and_sort_data()

    def _load_and_sort_data(self) -> List[NewsEntry]:
        # Load data from Excel and sort by publication time
        df = pd.read_excel(self.file_path)
        df['Время публикации'] = pd.to_datetime(df['Время публикации'], errors='coerce')
        df = df.dropna(subset=['Время публикации', 'Текст сообщения'])
        df = df.sort_values(by='Время публикации')
        messages = df.to_dict('records')

        return [
            NewsEntry(
                timestamp=record['Время публикации'].isoformat(),
                text=record['Текст сообщения'],
                url=record['Ссылка на сообщение']
            )
            for record in messages
        ]

    def get_entries(self):
        # Generator for iterating through the loaded messages
        yield from self.data


class ContentProcessor:
    def __init__(self, data_loader: NewsReader, text_pipeline: TextPipeline,
                 keyword_matcher: KeywordMatcher, cluster_handler: ClusterHandler,
                 notifier: Notifier, output_file: str) -> None:
        self.data_loader = data_loader
        self.text_pipeline = text_pipeline
        self.keyword_matcher = keyword_matcher
        self.cluster_handler = cluster_handler
        self.notifier = notifier
        self.spreadsheet_writer = SpreadsheetWriter(output_file)
        self.spreadsheet_writer.write_header(['Время публикации', 'Текст сообщения', 'Ссылка на сообщение', 'Кластер', 'Минцифры?'])

    def process_entries(self) -> None:
        for message in self.data_loader.get_entries():
            cleaned_text, embedding = self.text_pipeline.process_text(message['text'])

            # Classify relevance
            is_relevant = self.keyword_matcher.matches_keywords(cleaned_text)
            relevance_flag = 1 if is_relevant else 0

            # Check for uniqueness or assign a new cluster
            if not is_relevant:
                cluster_id = self.cluster_handler.assign_cluster_id(embedding)
            else:
                is_unique, cluster_id = self.cluster_handler.find_cluster(embedding)
                if is_unique:
                    cluster_id = self.cluster_handler.assign_cluster_id(embedding)
                    self.notifier.notify(message)

            # Write data to Excel
            self.spreadsheet_writer.append_row([message['timestamp'], message['text'], message['url'], cluster_id, relevance_flag])

        # Final save of the results
        self.spreadsheet_writer.save()

    def compute_cluster_significance(self):
        # Calculate cluster significance by reading the Excel file and updating it
        df = pd.read_excel(self.spreadsheet_writer.output_file)

        cluster_counts = df['Кластер'].value_counts()
        df['Значимость'] = df['Кластер'].map(cluster_counts)
        df = df.sort_values(by=['Значимость', 'Кластер', 'Время публикации'], ascending=[False, True, True])

        df.to_excel(self.spreadsheet_writer.output_file, index=False)
        print(f"Updated file with 'Значимость' column saved to {self.spreadsheet_writer.output_file}")


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

    data_loader = NewsReader(file_path)
    keyword_matcher = KeywordMatcher(keywords)
    cluster_handler = ClusterHandler()
    text_pipeline = TextPipeline()
    notifier = Notifier()

    content_processor = ContentProcessor(
        data_loader=data_loader,
        text_pipeline=text_pipeline,
        keyword_matcher=keyword_matcher,
        cluster_handler=cluster_handler,
        notifier=notifier,
        output_file=output_file
    )

    # Process all news entries
    content_processor.process_entries()

    # Calculate and add the "Significance" column
    content_processor.compute_cluster_significance()
