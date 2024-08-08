import streamlit as st
import tempfile
import pandas as pd
from news_monitoring import NewsMonitoringFactory

# Page settings
st.set_page_config(page_title="Мониторинг новостей", layout="centered")

# Title and description
st.title("Мониторинг новостей")
st.markdown(
    """
**Формат входного файла:**

- **Время публикации**: Дата и время публикации новости.
- **Текст сообщения**: Содержание новости.
- **Ссылка на сообщение**: URL-адрес новости.

**Формат выходного файла:**

- **Время публикации**: Дата и время публикации новости.
- **Текст сообщения**: Содержание новости.
- **Ссылка на сообщение**: URL-адрес новости.
- **Кластер**: Номер кластера, к которому относится новость.
- **Минцифры?**: Связана ли новость с Минцифры РФ, 0 или 1.
- **Значимость**: Количество дублей определённой новости.
"""
)

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
    output_file_path = input_file_path.replace(".xlsx", "_processed.xlsx")

    # Preview of the uploaded file
    st.subheader("Превью загруженного файла")
    df_uploaded = pd.read_excel(uploaded_file)
    st.dataframe(df_uploaded)

    # List of keywords
    keywords = [
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
    ]

    # Create the factory and news processor
    factory = NewsMonitoringFactory(
        input_path=input_file_path,
        keywords=keywords,
        pretrained_model="cointegrated/rubert-tiny2",
        output_path=output_file_path,
    )
    news_processor = factory.create_news_processor()

    # Progress bar
    progress_bar = st.progress(0)
    with st.spinner("Обработка новостей..."):
        news_processor.process_messages()
        progress_bar.progress(50)

        # Evaluate cluster significance
        news_processor.evaluate_clusters()
        progress_bar.progress(100)

    st.success("Обработка завершена! Вы можете скачать обработанный файл ниже.")

    # Update session state with the output file path
    st.session_state.output_file_path = output_file_path

    # Preview of the generated file
    st.subheader("Превью обработанного файла")
    df_processed = pd.read_excel(output_file_path)
    st.dataframe(df_processed.head(1000))

    # Button to download the processed file
    with open(output_file_path, "rb") as processed_file:
        st.download_button(
            label="Скачать обработанный файл",
            data=processed_file,
            file_name="обработанные_новости.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# If there is a generated file but a new file has not been uploaded
if st.session_state.output_file_path and not uploaded_file:
    st.subheader("Последний обработанный файл")
    df_last_processed = pd.read_excel(st.session_state.output_file_path)
    st.dataframe(df_last_processed.head(1000))

    # Button to download the last processed file
    with open(st.session_state.output_file_path, "rb") as processed_file:
        st.download_button(
            label="Скачать последний обработанный файл",
            data=processed_file,
            file_name="обработанные_новости.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
