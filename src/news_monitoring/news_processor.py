import datetime
import pandas as pd
from .news_loader import NewsLoader
from .message_processor import MessageProcessor
from .keyword_filter import KeywordFilter
from .cluster_processor import ClusterProcessor
from .streamlit_notifier import StreamlitNotifier


class NewsProcessor:
    def __init__(
        self,
        news_loader: NewsLoader,
        message_processor: MessageProcessor,
        relevance_filter: KeywordFilter,
        cluster_processor: ClusterProcessor,
        notifier: StreamlitNotifier,
        progress_callback=None,
    ) -> None:
        self.news_loader = news_loader
        self.message_processor = message_processor
        self.relevance_filter = relevance_filter
        self.cluster_processor = cluster_processor
        self.notifier = notifier
        self.progress_callback = progress_callback

        self.processed_records = []

    def process_messages(self) -> None:
        total_messages = len(self.news_loader.news_data)
        for index, news_message in enumerate(self.news_loader.get_entries()):
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
            is_notified = is_relevant and is_unique
            if is_notified:
                self.notifier.notify(news_message)

            # Save the processed record
            self.processed_records.append(
                {
                    "published_at": news_message["published_at"],
                    "content": news_message["content"],
                    "url": news_message["url"],
                    "cluster": assigned_cluster,
                    "relevant": relevance_flag,
                    "notified": is_notified,
                }
            )

            # Update progress
            if self.progress_callback:
                # time + percentage
                time = datetime.datetime.fromisoformat(
                    news_message["published_at"]
                ).strftime("%Y-%m-%d %H:%M")
                ratio = (index + 1) / total_messages
                comment = f"{time} - {ratio:.1%}"
                self.progress_callback(ratio, comment)

    def evaluate_clusters(self):
        # Calculate cluster significance
        news_df = pd.DataFrame(self.processed_records)

        cluster_sizes = news_df["cluster"].value_counts()
        news_df["significance"] = news_df["cluster"].map(cluster_sizes)
        news_df = news_df.sort_values(
            by=["significance", "cluster", "published_at"],
            ascending=[False, True, True],
        )

        return news_df[news_df["notified"] == True]
