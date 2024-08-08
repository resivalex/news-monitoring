from .types import NewsMessage


class Notifier:
    def notify(self, news_message: NewsMessage) -> None:
        # Print notification for a new unique event
        print("-------- Notification --------")
        print(f"Time: {news_message['published_at']}")
        print(f"URL: {news_message['url']}")
        print(news_message["content"])
        print("------------------------------")
