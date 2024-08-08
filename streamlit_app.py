import streamlit as st
import tempfile
import os
from news_monitoring import NewsMonitoringFactory

# Настройки страницы
st.set_page_config(page_title="Мониторинг новостей", layout="centered")

# Заголовок и описание
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

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл в формате XLSX", type=["xlsx"])

if uploaded_file is not None:
    st.write("Файл загружен")

    # Сохранение загруженного файла во временный каталог
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_file_path = tmp_file.name

    # Путь для сохранения выходного файла
    output_file_path = input_file_path.replace(".xlsx", "_processed.xlsx")

    # Список ключевых слов
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

    # Создание фабрики и обработчика новостей
    factory = NewsMonitoringFactory(
        input_path=input_file_path,
        keywords=keywords,
        pretrained_model="cointegrated/rubert-tiny2",
        output_path=output_file_path,
    )
    news_processor = factory.create_news_processor()

    # Прогресс-бар
    progress_bar = st.progress(0)
    with st.spinner("Обработка новостей..."):
        news_processor.process_messages()
        progress_bar.progress(50)

        # Вычисление значимости кластеров
        news_processor.evaluate_clusters()
        progress_bar.progress(100)

    st.success("Обработка завершена! Вы можете скачать обработанный файл ниже.")

    # Кнопка для загрузки обработанного файла
    with open(output_file_path, "rb") as processed_file:
        st.download_button(
            label="Скачать обработанный файл",
            data=processed_file,
            file_name="обработанные_новости.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Очистка временных файлов
    os.remove(input_file_path)
    os.remove(output_file_path)
