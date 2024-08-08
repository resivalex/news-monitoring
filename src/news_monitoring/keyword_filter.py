class KeywordFilter:
    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords

    def has_relevant_terms(self, text: str) -> bool:
        # Check if the text contains any of the relevant keywords
        lowercased_text = text.lower()
        return any(keyword.lower() in lowercased_text for keyword in self.keywords)
