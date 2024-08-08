import torch
from transformers import AutoTokenizer, AutoModel
from .types import MessageEmbedding


class MessageProcessor:
    def __init__(self, pretrained_model: str) -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embedding_model = AutoModel.from_pretrained(pretrained_model)

    def clean_and_vectorize(self, text: str) -> (str, MessageEmbedding):
        cleaned_text = self._clean_text(text)
        text_embedding = self._generate_embedding(cleaned_text)
        return cleaned_text, text_embedding

    def _clean_text(self, text: str) -> str:
        # Clean and preprocess the text
        return " ".join(text.lower().split())

    def _generate_embedding(self, text: str) -> MessageEmbedding:
        # Generate an embedding for the text using a pre-trained model
        tokenized_text = self.text_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            model_outputs = self.embedding_model(**tokenized_text)

        text_embeddings = model_outputs.last_hidden_state[:, 0, :].numpy()
        return MessageEmbedding(embedding=text_embeddings[0])
