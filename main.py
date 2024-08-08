from news_monitoring import NewsMonitoringFactory

if __name__ == "__main__":
    file_path = "./data/posts_mc.xlsx"
    output_file = "./data/clustered.xlsx"

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

    factory = NewsMonitoringFactory(file_path, output_file, keywords)
    news_processor = factory.create_news_processor()

    # Process all news entries
    news_processor.process_messages()

    # Calculate and add the "Significance" column
    news_processor.evaluate_clusters()
