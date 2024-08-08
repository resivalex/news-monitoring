import pandas as pd
from .news_loader import NewsLoader
from .message_processor import MessageProcessor
from .keyword_filter import KeywordFilter
from .cluster_processor import ClusterProcessor
from .notifier import Notifier
from .excel_exporter import ExcelExporter


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

            # Check for uniqueness and assign cluster
            is_unique, assigned_cluster = self.cluster_processor.process_embedding(
                text_embedding
            )

            # Notify if the message is relevant and unique
            if is_relevant and is_unique:
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
