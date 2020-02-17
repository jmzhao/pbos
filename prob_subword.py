import logging

from utils.load import load_embedding
from utils.preprocess import count_subwords, normalize_prob
from trainer import Trainer

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    vocab, embeddings = load_embedding("datasets/GoogleNews-vectors-negative300.pkl")

    subword_count = count_subwords(vocab)
    subword_score = normalize_prob(subword_count)
    trainer = Trainer(subword_score, embedding_dim=embeddings[0].shape[0])

    for word, embedding in zip(vocab, embeddings):
        trainer.train(word, embedding)
