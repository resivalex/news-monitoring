from .news_loader import NewsLoader
from .keyword_filter import KeywordFilter
from .cluster_processor import ClusterProcessor
from .message_processor import MessageProcessor
from .notifier import Notifier
from .news_processor import NewsProcessor


class NewsMonitoringFactory:
    def __init__(self, file_path: str, output_file: str, keywords: list[str]) -> None:
        self.file_path = file_path
        self.output_file = output_file
        self.keywords = keywords

    def create_news_processor(self) -> NewsProcessor:
        news_loader = NewsLoader(self.file_path)
        relevance_filter = KeywordFilter(self.keywords)
        cluster_processor = ClusterProcessor()
        message_processor = MessageProcessor()
        notifier = Notifier()

        return NewsProcessor(
            news_loader=news_loader,
            message_processor=message_processor,
            relevance_filter=relevance_filter,
            cluster_processor=cluster_processor,
            notifier=notifier,
            output_file=self.output_file,
        )
