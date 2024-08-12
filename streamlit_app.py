import streamlit as st
import tempfile
import pandas as pd
from news_monitoring import NewsMonitoringFactory


def process_news(input_file_path, update_progress):
    factory = NewsMonitoringFactory(
        input_path=input_file_path,
        keywords=[
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
        ],
        pretrained_model="cointegrated/rubert-tiny2",
        progress_callback=update_progress,
    )

    news_processor = factory.create_news_processor()

    # Process news
    news_processor.process_messages()

    # Evaluate cluster significance
    news_processor.evaluate_clusters()


# Page settings
st.set_page_config(page_title="Мониторинг новостей", layout="centered")

# File upload
with st.sidebar:
    st.markdown(open("readme.md").read())

    uploaded_file = st.file_uploader("Загрузите файл в формате XLSX", type=["xlsx"])

if uploaded_file is not None:
    with st.sidebar:
        st.write("Файл загружен!")

    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_file_path = tmp_file.name

    with st.sidebar:
        placeholder = st.empty()
        progress_bar = st.progress(0)

        def update_progress(progress, comment):
            progress_bar.progress(int(progress * 100))
            placeholder.text(comment)

    st.chat_input(
        "",
        disabled=True,
        key="user_message_chat_input",
    )

    process_news(input_file_path, update_progress)
