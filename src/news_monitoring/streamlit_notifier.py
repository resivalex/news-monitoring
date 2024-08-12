from .types import NewsMessage
import streamlit as st
import datetime


class StreamlitNotifier:
    def notify(self, news_message: NewsMessage) -> None:
        with st.chat_message("ai"):
            time = datetime.datetime.fromisoformat(
                news_message["published_at"]
            ).strftime("%Y-%m-%d %H:%M")
            with st.expander(f"{time}\n\n{news_message['content'][0:200]}..."):
                st.write(news_message["url"] + "\n\n")
                st.text(news_message["content"])
