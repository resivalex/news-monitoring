from openpyxl import Workbook
from typing import List, Optional, TypedDict, Any
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


# TypedDict for representing a news message structure
class NewsMessage(TypedDict):
    published_at: str
    content: str
    url: str


class MessageEmbedding(TypedDict):
    embedding: np.ndarray


class ClusterProcessor:
    def __init__(self) -> None:
        self.stored_embeddings = []
        self.cluster_counter = 1
        self.cluster_ids = []

    def assign_cluster(self, text_embedding: MessageEmbedding) -> int:
        # Save the embedding and assign a new cluster ID
        self.stored_embeddings.append(text_embedding)
        assigned_cluster = self.cluster_counter
        self.cluster_ids.append(assigned_cluster)
        self.cluster_counter += 1
        return assigned_cluster

    def find_cluster(
        self, text_embedding: MessageEmbedding, similarity_threshold: float = 0.9
    ) -> (bool, Optional[int]):
        # Check if the embedding is unique based on cosine similarity
        for i, stored_vector in enumerate(self.stored_embeddings):
            cosine_similarity = self._calculate_similarity(
                text_embedding["embedding"], stored_vector["embedding"]
            )
            if cosine_similarity >= similarity_threshold:
                return False, self.cluster_ids[i]
        return True, None

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Calculate cosine similarity between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)


class KeywordFilter:
    def __init__(self, keywords: List[str]) -> None:
        self.keywords = keywords

    def has_relevant_terms(self, text: str) -> bool:
        # Check if the text contains any of the relevant keywords
        lowercased_text = text.lower()
        return any(keyword.lower() in lowercased_text for keyword in self.keywords)


class ExcelExporter:
    def __init__(self, file_path: str, save_interval: int = 1000):
        self.file_path = file_path
        self.workbook = Workbook()
        self.worksheet = self.workbook.active
        self.save_interval = save_interval
        self.rows_written = 0

    def write_header(self, columns: List[str]):
        self.worksheet.append(columns)
        self.rows_written += 1

    def write_row(self, row: List[Any]):
        # Append a row to the Excel file and save periodically
        self.worksheet.append(row)
        self.rows_written += 1

        if self.rows_written % self.save_interval == 0:
            self.save()
            print(f"Auto-saved after {self.rows_written} rows.")

    def save(self):
        # Save the Excel file
        self.workbook.save(self.file_path)
        print(f"File saved to {self.file_path}")


class MessageProcessor:
    def __init__(self) -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.embedding_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def clean_and_vectorize(self, text: str) -> (str, MessageEmbedding):
        cleaned_text = self._clean_text(text)
        text_embedding = self._generate_embedding(cleaned_text)
        return cleaned_text, text_embedding

    def _clean_text(self, text: str) -> str:
        # Clean and preprocess the text
        return " ".join(text.lower().split())

    def _generate_embedding(self, text: str) -> MessageEmbedding:
        # Generate an embedding for the text using a pre-trained model
        tokenized_text = self.text_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            model_outputs = self.embedding_model(**tokenized_text)

        text_embeddings = model_outputs.last_hidden_state[:, 0, :].numpy()
        return MessageEmbedding(embedding=text_embeddings[0])


class Notifier:
    def notify(self, news_message: NewsMessage) -> None:
        # Print notification for a new unique event
        print("-------- Notification --------")
        print(f"Time: {news_message['published_at']}")
        print(f"URL: {news_message['url']}")
        print(news_message["content"])
        print("------------------------------")


class NewsLoader:
    def __init__(self, data_file: str) -> None:
        self.data_file = data_file
        self.news_data = self._load_and_sort()

    def _load_and_sort(self) -> List[NewsMessage]:
        # Load data from Excel and sort by publication time
        news_df = pd.read_excel(self.data_file)
        news_df["Время публикации"] = pd.to_datetime(
            news_df["Время публикации"], errors="coerce"
        )
        news_df = news_df.dropna(subset=["Время публикации", "Текст сообщения"])
        news_df = news_df.sort_values(by="Время публикации")
        news_records = news_df.to_dict("records")

        return [
            NewsMessage(
                published_at=record["Время публикации"].isoformat(),
                content=record["Текст сообщения"],
                url=record["Ссылка на сообщение"],
            )
            for record in news_records
        ]

    def get_entries(self):
        # Generator for iterating through the loaded messages
        yield from self.news_data


class NewsProcessor:
    def __init__(
        self,
        news_loader: NewsLoader,
        message_processor: MessageProcessor,
        relevance_filter: KeywordFilter,
        cluster_processor: ClusterProcessor,
        notifier: Notifier,
        output_file: str,
    ) -> None:
        self.news_loader = news_loader
        self.message_processor = message_processor
        self.relevance_filter = relevance_filter
        self.cluster_processor = cluster_processor
        self.notifier = notifier
        self.excel_exporter = ExcelExporter(output_file)
        self.excel_exporter.write_header(
            [
                "Время публикации",
                "Текст сообщения",
                "Ссылка на сообщение",
                "Кластер",
                "Минцифры?",
            ]
        )

    def process_messages(self) -> None:
        for news_message in self.news_loader.get_entries():
            cleaned_text, text_embedding = self.message_processor.clean_and_vectorize(
                news_message["content"]
            )

            # Classify relevance
            is_relevant = self.relevance_filter.has_relevant_terms(cleaned_text)
            relevance_flag = 1 if is_relevant else 0

            # Check for uniqueness or assign a new cluster
            if not is_relevant:
                assigned_cluster = self.cluster_processor.assign_cluster(text_embedding)
            else:
                is_unique, assigned_cluster = self.cluster_processor.find_cluster(
                    text_embedding
                )
                if is_unique:
                    assigned_cluster = self.cluster_processor.assign_cluster(
                        text_embedding
                    )
                    self.notifier.notify(news_message)

            # Write data to Excel
            self.excel_exporter.write_row(
                [
                    news_message["published_at"],
                    news_message["content"],
                    news_message["url"],
                    assigned_cluster,
                    relevance_flag,
                ]
            )

        # Final save of the results
        self.excel_exporter.save()

    def evaluate_clusters(self):
        # Calculate cluster significance by reading the Excel file and updating it
        news_df = pd.read_excel(self.excel_exporter.file_path)

        cluster_sizes = news_df["Кластер"].value_counts()
        news_df["Значимость"] = news_df["Кластер"].map(cluster_sizes)
        news_df = news_df.sort_values(
            by=["Значимость", "Кластер", "Время публикации"],
            ascending=[False, True, True],
        )

        news_df.to_excel(self.excel_exporter.file_path, index=False)
        print(
            f"Updated file with 'Значимость' column saved to {self.excel_exporter.file_path}"
        )


if __name__ == "__main__":
    file_path = "./data/posts_mc.xlsx"
    output_file = "./data/clustered.xlsx"

    keywords = [
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

    news_loader = NewsLoader(file_path)
    relevance_filter = KeywordFilter(keywords)
    cluster_processor = ClusterProcessor()
    message_processor = MessageProcessor()
    notifier = Notifier()

    news_processor = NewsProcessor(
        news_loader=news_loader,
        message_processor=message_processor,
        relevance_filter=relevance_filter,
        cluster_processor=cluster_processor,
        notifier=notifier,
        output_file=output_file,
    )

    # Process all news entries
    news_processor.process_messages()

    # Calculate and add the "Significance" column
    news_processor.evaluate_clusters()
