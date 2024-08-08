import streamlit as st
import tempfile
import os
import pandas as pd
from news_monitoring import NewsMonitoringFactory


def process_news(input_file_path, output_file_path, update_progress):
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
        output_path=output_file_path,
        progress_callback=update_progress,
    )

    news_processor = factory.create_news_processor()

    # Process news
    news_processor.process_messages()

    # Evaluate cluster significance
    news_processor.evaluate_clusters()


# Page settings
st.set_page_config(page_title="Мониторинг новостей", layout="centered")

st.markdown(open("readme.md").read())

# State to store the path to the generated file
if "output_file_path" not in st.session_state:
    st.session_state.output_file_path = None

# File upload
uploaded_file = st.file_uploader("Загрузите файл в формате XLSX", type=["xlsx"])

if uploaded_file is not None:
    st.write("Файл загружен!")

    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_file_path = tmp_file.name

    # Path for saving the output file
    output_file_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    output_file_path = os.path.join(
        tempfile.gettempdir(), f"{output_file_name}_processed.xlsx"
    )

    # Preview of the uploaded file
    st.subheader("Превью загруженного файла")
    df_uploaded = pd.read_excel(uploaded_file)
    st.dataframe(df_uploaded.head(100))

    progress_bar = st.progress(0)

    def update_progress(progress):
        progress_bar.progress(int(progress * 100))

    # Process news in the background
    with st.spinner("Обработка новостей..."):
        process_news(input_file_path, output_file_path, update_progress)
        progress_bar.progress(100)

    st.success("Обработка завершена! Вы можете скачать обработанный файл ниже.")

    # Update session state with the output file path
    st.session_state.output_file_path = output_file_path

    # Preview of the generated file
    st.subheader("Превью обработанного файла")
    df_processed = pd.read_excel(output_file_path)
    st.dataframe(df_processed.head(100))

    # Button to download the processed file
    with open(output_file_path, "rb") as processed_file:
        st.download_button(
            label="Скачать обработанный файл",
            data=processed_file,
            file_name=f"{output_file_name}_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
