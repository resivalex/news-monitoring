import pandas as pd
from .types import NewsMessage
from typing import List


class NewsLoader:
    def __init__(self, data_file: str) -> None:
        self.data_file = data_file
        self.news_data = self._load_and_sort()

    def _load_and_sort(self) -> List[NewsMessage]:
        # Load data from Excel and sort by publication time
        news_df = pd.read_excel(self.data_file)
        news_df["Время публикации"] = pd.to_datetime(
            news_df["Время публикации"], format="%d.%m.%Y %H:%M"
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
