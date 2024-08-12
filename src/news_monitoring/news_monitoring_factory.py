from .news_loader import NewsLoader
from .keyword_filter import KeywordFilter
from .cluster_processor import ClusterProcessor
from .message_processor import MessageProcessor
from .streamlit_notifier import StreamlitNotifier
from .news_processor import NewsProcessor


class NewsMonitoringFactory:
    def __init__(
        self,
        input_path: str,
        keywords: list[str],
        pretrained_model: str,
        progress_callback=None,
    ) -> None:
        self.input_path = input_path
        self.keywords = keywords
        self.pretrained_model = pretrained_model
        self.progress_callback = progress_callback

    def create_news_processor(self) -> NewsProcessor:
        news_loader = NewsLoader(self.input_path)
        relevance_filter = KeywordFilter(self.keywords)
        cluster_processor = ClusterProcessor()
        message_processor = MessageProcessor(pretrained_model=self.pretrained_model)
        notifier = StreamlitNotifier()

        return NewsProcessor(
            news_loader=news_loader,
            message_processor=message_processor,
            relevance_filter=relevance_filter,
            cluster_processor=cluster_processor,
            notifier=notifier,
            progress_callback=self.progress_callback,
        )
