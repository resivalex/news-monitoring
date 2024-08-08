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

    def assign_cluster_id(self, text_vector: TextVector) -> int:
        # Save the embedding and assign a new cluster ID
        self.embeddings.append(text_vector)
        cluster_id = self.cluster_count
        self.cluster_ids.append(cluster_id)
        self.cluster_count += 1
        return cluster_id

    def find_cluster(self, text_vector: TextVector, similarity_threshold: float = 0.9) -> (bool, Optional[int]):
        # Check if the embedding is unique based on cosine similarity
        for i, stored_vector in enumerate(self.embeddings):
            cosine_similarity = self._cosine_similarity(text_vector['vector'], stored_vector['vector'])
            if cosine_similarity >= similarity_threshold:
                return False, self.cluster_ids[i]
        return True, None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Calculate cosine similarity between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)


class KeywordMatcher:
    def __init__(self, key_terms: List[str]) -> None:
        self.key_terms = key_terms

    def matches_keywords(self, text: str) -> bool:
        # Check if the text contains any of the relevant keywords
        lowercased_text = text.lower()
        return any(keyword.lower() in lowercased_text for keyword in self.key_terms)


class SpreadsheetWriter:
    def __init__(self, output_path: str, save_interval: int = 1000):
        self.output_path = output_path
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
        self.workbook.save(self.output_path)
        print(f"File saved to {self.output_path}")


class TextPipeline:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def clean_text(self, text: str) -> str:
        # Clean and preprocess the text
        return " ".join(text.lower().split())

    def vectorize_text(self, text: str) -> TextVector:
        # Generate an embedding for the text using a pre-trained model
        tokenized_text = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            model_outputs = self.model(**tokenized_text)

        text_embeddings = model_outputs.last_hidden_state[:, 0, :].numpy()
        return TextVector(vector=text_embeddings[0])

    def process_text(self, text: str) -> (str, TextVector):
        cleaned_text = self.clean_text(text)
        text_vector = self.vectorize_text(cleaned_text)
        return cleaned_text, text_vector


class Notifier:
    def notify(self, news_entry: NewsEntry) -> None:
        # Print notification for a new unique event
        print("-------- Notification --------")
        print(f"Time: {news_entry['timestamp']}")
        print(f"URL: {news_entry['url']}")
        print(news_entry["text"])
        print("------------------------------")


class NewsReader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_and_sort_data()

    def _load_and_sort_data(self) -> List[NewsEntry]:
        # Load data from Excel and sort by publication time
        news_df = pd.read_excel(self.file_path)
        news_df['Время публикации'] = pd.to_datetime(news_df['Время публикации'], errors='coerce')
        news_df = news_df.dropna(subset=['Время публикации', 'Текст сообщения'])
        news_df = news_df.sort_values(by='Время публикации')
        news_entries = news_df.to_dict('records')

        return [
            NewsEntry(
                timestamp=entry['Время публикации'].isoformat(),
                text=entry['Текст сообщения'],
                url=entry['Ссылка на сообщение']
            )
            for entry in news_entries
        ]

    def get_entries(self):
        # Generator for iterating through the loaded messages
        yield from self.data


class ContentProcessor:
    def __init__(self, data_loader: NewsReader, text_pipeline: TextPipeline,
                 keyword_matcher: KeywordMatcher, cluster_handler: ClusterHandler,
                 notifier: Notifier, output_path: str) -> None:
        self.data_loader = data_loader
        self.text_pipeline = text_pipeline
        self.keyword_matcher = keyword_matcher
        self.cluster_handler = cluster_handler
        self.notifier = notifier
        self.spreadsheet_writer = SpreadsheetWriter(output_path)
        self.spreadsheet_writer.write_header(['Время публикации', 'Текст сообщения', 'Ссылка на сообщение', 'Кластер', 'Минцифры?'])

    def process_entries(self) -> None:
        for news_entry in self.data_loader.get_entries():
            cleaned_text, text_vector = self.text_pipeline.process_text(news_entry['text'])

            # Classify relevance
            is_relevant = self.keyword_matcher.matches_keywords(cleaned_text)
            relevance_flag = 1 if is_relevant else 0

            # Check for uniqueness or assign a new cluster
            if not is_relevant:
                cluster_id = self.cluster_handler.assign_cluster_id(text_vector)
            else:
                is_unique, cluster_id = self.cluster_handler.find_cluster(text_vector)
                if is_unique:
                    cluster_id = self.cluster_handler.assign_cluster_id(text_vector)
                    self.notifier.notify(news_entry)

            # Write data to Excel
            self.spreadsheet_writer.append_row([news_entry['timestamp'], news_entry['text'], news_entry['url'], cluster_id, relevance_flag])

        # Final save of the results
        self.spreadsheet_writer.save()

    def compute_cluster_significance(self):
        # Calculate cluster significance by reading the Excel file and updating it
        news_df = pd.read_excel(self.spreadsheet_writer.output_path)

        cluster_sizes = news_df['Кластер'].value_counts()
        news_df['Значимость'] = news_df['Кластер'].map(cluster_sizes)
        news_df = news_df.sort_values(by=['Значимость', 'Кластер', 'Время публикации'], ascending=[False, True, True])

        news_df.to_excel(self.spreadsheet_writer.output_path, index=False)
        print(f"Updated file with 'Значимость' column saved to {self.spreadsheet_writer.output_path}")


if __name__ == "__main__":
    news_file_path = "./data/posts_mc.xlsx"
    output_path = "./data/clustered.xlsx"

    key_terms = [
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

    data_loader = NewsReader(news_file_path)
    keyword_matcher = KeywordMatcher(key_terms)
    cluster_handler = ClusterHandler()
    text_pipeline = TextPipeline()
    notifier = Notifier()

    content_processor = ContentProcessor(
        data_loader=data_loader,
        text_pipeline=text_pipeline,
        keyword_matcher=keyword_matcher,
        cluster_handler=cluster_handler,
        notifier=notifier,
        output_path=output_path
    )

    # Process all news entries
    content_processor.process_entries()

    # Calculate and add the "Significance" column
    content_processor.compute_cluster_significance()
