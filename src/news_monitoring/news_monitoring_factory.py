from .news_loader import NewsLoader
from .keyword_filter import KeywordFilter
from .cluster_processor import ClusterProcessor
from .message_processor import MessageProcessor
from .notifier import Notifier
from .news_processor import NewsProcessor


class NewsMonitoringFactory:
    def __init__(self, input_path: str, output_path: str, keywords: list[str]) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.keywords = keywords

    def create_news_processor(self) -> NewsProcessor:
        news_loader = NewsLoader(self.input_path)
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
            output_path=self.output_path,
        )
