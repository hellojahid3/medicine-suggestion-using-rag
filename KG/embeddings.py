from config import get_ollama_embeddings

embedding_model = None

def get_embedding_model_instance():
    global embedding_model
    if embedding_model is None:
        embedding_model = get_ollama_embeddings()
    return embedding_model

def get_text_embedding(text: str):
    """Generates embedding for a given text."""
    model = get_embedding_model_instance()
    return model.embed_query(text)

def get_texts_embeddings(texts: list[str]):
    """Generates embeddings for a list of texts."""
    model = get_embedding_model_instance()
    return model.embed_documents(texts)
