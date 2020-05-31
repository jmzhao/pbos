import logging

logger = logging.getLogger(__name__)


def _is_word(w):
    return w.isalpha() and w.isascii() and w.islower()


def clean_target_emb(raw_vocab, raw_emb):
    logger.info("normalizing...")

    vocab, emb = [], []
    for w, e in zip(raw_vocab, raw_emb):
        if _is_word(w):
            vocab.append(w)
            emb.append(e)
    return vocab, emb
