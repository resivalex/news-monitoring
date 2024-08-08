from openpyxl import Workbook
from typing import List, Optional, TypedDict, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


# TypedDict for representing a news record structure
class NewsRecord(TypedDict):
    timestamp: str
    text: str
    url: str


class TextEmbedding(TypedDict):
    vector: np.ndarray


class ClusterManager:
    def __init__(self) -> None:
        self.embeddings = []
        self.cluster_count = 1
        self.cluster_ids = []

    def assign_cluster_id(self, text_embedding: TextEmbedding) -> int:
        # Save the embedding and assign a new cluster ID
        self.embeddings.append(text_embedding)
        cluster_id = self.cluster_count
        self.cluster_ids.append(cluster_id)
        self.cluster_count += 1
        return cluster_id

    def locate_cluster(
        self, text_embedding: TextEmbedding, similarity_threshold: float = 0.9
    ) -> (bool, Optional[int]):
        # Check if the embedding is unique based on cosine similarity
        for i, stored_vector in enumerate(self.embeddings):
            cosine_similarity = self._cosine_similarity(
                text_embedding["vector"], stored_vector["vector"]
            )
            if cosine_similarity >= similarity_threshold:
                return False, self.cluster_ids[i]
        return True, None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Calculate cosine similarity between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)


class KeywordDetector:
    def __init__(self, key_terms: List[str]) -> None:
        self.key_terms = key_terms

    def contains_keywords(self, text: str) -> bool:
        # Check if the text contains any of the relevant keywords
        lowercased_text = text.lower()
        return any(keyword.lower() in lowercased_text for keyword in self.key_terms)


class ExcelFileWriter:
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


class TextProcessor:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def preprocess_text(self, text: str) -> str:
        # Clean and preprocess the text
        return " ".join(text.lower().split())

    def generate_embedding(self, text: str) -> TextEmbedding:
        # Generate an embedding for the text using a pre-trained model
        tokenized_text = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            model_outputs = self.model(**tokenized_text)

        text_embeddings = model_outputs.last_hidden_state[:, 0, :].numpy()
        return TextEmbedding(vector=text_embeddings[0])

    def process_and_embed(self, text: str) -> (str, TextEmbedding):
        cleaned_text = self.preprocess_text(text)
        text_embedding = self.generate_embedding(cleaned_text)
        return cleaned_text, text_embedding


class Notifier:
    def send_notification(self, news_record: NewsRecord) -> None:
        # Print notification for a new unique event
        print("-------- Notification --------")
        print(f"Time: {news_record['timestamp']}")
        print(f"URL: {news_record['url']}")
        print(news_record["text"])
        print("------------------------------")


class NewsDataLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_and_sort_data()

    def _load_and_sort_data(self) -> List[NewsRecord]:
        # Load data from Excel and sort by publication time
        news_df = pd.read_excel(self.file_path)
        news_df["Время публикации"] = pd.to_datetime(
            news_df["Время публикации"], errors="coerce"
        )
        news_df = news_df.dropna(subset=["Время публикации", "Текст сообщения"])
        news_df = news_df.sort_values(by="Время публикации")
        news_records = news_df.to_dict("records")

        return [
            NewsRecord(
                timestamp=record["Время публикации"].isoformat(),
                text=record["Текст сообщения"],
                url=record["Ссылка на сообщение"],
            )
            for record in news_records
        ]

    def load_entries(self):
        # Generator for iterating through the loaded messages
        yield from self.data


class EntryProcessor:
    def __init__(
        self,
        data_loader: NewsDataLoader,
        text_processor: TextProcessor,
        keyword_detector: KeywordDetector,
        cluster_manager: ClusterManager,
        notifier: Notifier,
        output_path: str,
    ) -> None:
        self.data_loader = data_loader
        self.text_processor = text_processor
        self.keyword_detector = keyword_detector
        self.cluster_manager = cluster_manager
        self.notifier = notifier
        self.excel_file_writer = ExcelFileWriter(output_path)
        self.excel_file_writer.write_header(
            [
                "Время публикации",
                "Текст сообщения",
                "Ссылка на сообщение",
                "Кластер",
                "Минцифры?",
            ]
        )

    def process_all_entries(self) -> None:
        for news_record in self.data_loader.load_entries():
            cleaned_text, text_embedding = self.text_processor.process_and_embed(
                news_record["text"]
            )

            # Classify relevance
            is_relevant = self.keyword_detector.contains_keywords(cleaned_text)
            relevance_flag = 1 if is_relevant else 0

            # Check for uniqueness or assign a new cluster
            if not is_relevant:
                cluster_id = self.cluster_manager.assign_cluster_id(text_embedding)
            else:
                is_unique, cluster_id = self.cluster_manager.locate_cluster(
                    text_embedding
                )
                if is_unique:
                    cluster_id = self.cluster_manager.assign_cluster_id(text_embedding)
                    self.notifier.send_notification(news_record)

            # Write data to Excel
            self.excel_file_writer.append_row(
                [
                    news_record["timestamp"],
                    news_record["text"],
                    news_record["url"],
                    cluster_id,
                    relevance_flag,
                ]
            )

        # Final save of the results
        self.excel_file_writer.save()

    def calculate_cluster_significance(self):
        # Calculate cluster significance by reading the Excel file and updating it
        news_df = pd.read_excel(self.excel_file_writer.output_path)

        cluster_sizes = news_df["Кластер"].value_counts()
        news_df["Значимость"] = news_df["Кластер"].map(cluster_sizes)
        news_df = news_df.sort_values(
            by=["Значимость", "Кластер", "Время публикации"],
            ascending=[False, True, True],
        )

        news_df.to_excel(self.excel_file_writer.output_path, index=False)
        print(
            f"Updated file with 'Значимость' column saved to {self.excel_file_writer.output_path}"
        )


if __name__ == "__main__":
    news_file_path = "./data/posts_mc.xlsx"
    output_path = "./data/clustered.xlsx"

    key_terms = [
        "минцифры",
        "минцифра",
        "минцифре",
        "минцифрой",
        "минцифрах",
        "министерство цифрового развития",
        "министерству цифрового развития",
        "министерства цифрового развития",
        "министерством цифрового развития",
        "министерство цифровизации",
        "министерству цифровизации",
        "министерства цифровизации",
        "министерством цифровизации",
        "цифровое министерство",
        "цифрового министерства",
        "цифровому министерству",
        "цифровым министерством",
        "министерство цифровой экономики",
        "министерству цифровой экономики",
        "министерства цифровой экономики",
        "министерством цифровой экономики",
        "министерство цифровых технологий",
        "министерству цифровых технологий",
        "министерства цифровых технологий",
        "министерством цифровых технологий",
    ]

    data_loader = NewsDataLoader(news_file_path)
    keyword_detector = KeywordDetector(key_terms)
    cluster_manager = ClusterManager()
    text_processor = TextProcessor()
    notifier = Notifier()

    entry_processor = EntryProcessor(
        data_loader=data_loader,
        text_processor=text_processor,
        keyword_detector=keyword_detector,
        cluster_manager=cluster_manager,
        notifier=notifier,
        output_path=output_path,
    )

    # Process all news entries
    entry_processor.process_all_entries()

    # Calculate and add the "Significance" column
    entry_processor.calculate_cluster_significance()
