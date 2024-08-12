from .types import NewsMessage
import streamlit as st


class StreamlitNotifier:
    def notify(self, news_message: NewsMessage) -> None:
        with st.expander(
            f"{news_message['published_at']} {news_message['content'][0:200]}..."
        ):
            st.write(news_message["url"])
            st.text(news_message["content"])
