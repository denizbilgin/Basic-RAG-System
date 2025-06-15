from typing import List, AnyStr
from sentence_transformers import SentenceTransformer
from torch import Tensor


class Embedder:
    def __init__(self, model_name: AnyStr = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        print(f"Embedding model '{model_name}' loaded.")

    def embed_documents(self, documents: List[AnyStr]) -> Tensor:
        document_embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)

        print(f"Generated embeddings for {len(document_embeddings)} documents.")
        print(f"Each embedding has a dimension of: {document_embeddings.shape[1]}")
        return document_embeddings

