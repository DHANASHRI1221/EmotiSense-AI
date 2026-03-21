from sentence_transformers import SentenceTransformer

class PreprocessorBERT:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def fit_transform(self, texts):
        texts = [t if t is not None else "" for t in texts]
        return self.model.encode(texts)

    def transform(self, texts):
        texts = [t if t is not None else "" for t in texts]
        return self.model.encode(texts)