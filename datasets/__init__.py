from .target_vectors import prepare_target_vector_paths
from .ud import prepare_ud_paths
from .word_freq.polyglot import prepare_polyglot_freq_paths
from .word_freq.unigram import prepare_unigram_freq_paths
from .word_similarity import prepare_ws_combined_query_path, prepare_ws_dataset_paths, get_ws_dataset_names

polyglot_languages = [
    'ar', 'bg', 'cs', 'da', 'el', 'en', 'es', 'eu', 'fa', 'he', 'hi', 'hu',
    'id', 'it', 'kk', 'lv', 'ro', 'ru', 'sv', 'ta', 'tr', 'vi', 'zh',
]

__all__ = [
    "polyglot_languages",

    "prepare_target_vector_paths",

    "prepare_ws_combined_query_path",
    "prepare_ws_dataset_paths",
    "get_ws_dataset_names",

    "prepare_ud_paths",

    "prepare_polyglot_freq_paths",
    "prepare_unigram_freq_paths"

]
