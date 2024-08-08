from news_monitoring import (
    NewsLoader,
    KeywordFilter,
    ClusterProcessor,
    MessageProcessor,
    Notifier,
    NewsProcessor,
)


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

    news_loader = NewsLoader(file_path)
    relevance_filter = KeywordFilter(keywords)
    cluster_processor = ClusterProcessor()
    message_processor = MessageProcessor()
    notifier = Notifier()

    news_processor = NewsProcessor(
        news_loader=news_loader,
        message_processor=message_processor,
        relevance_filter=relevance_filter,
        cluster_processor=cluster_processor,
        notifier=notifier,
        output_file=output_file,
    )

    # Process all news entries
    news_processor.process_messages()

    # Calculate and add the "Significance" column
    news_processor.evaluate_clusters()
