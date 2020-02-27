import os

languages = [
    "ar",
    "bg",
    "cs",
    "da",
    "el",
    "en",
    "es",
    "eu",
    "fa",
    "he",
    "hi",
    "hu",
    "id",
    "it",
    "kk",
    "lv",
    "ro",
    "ru",
    "sv",
    "ta",
    "tr",
    "vi",
    "zh",
]

language_code_to_folder = {
    "cu": "UD_Old_Church_Slavonic",
    "id": "UD_Indonesian",
    "ug": "UD_Uyghur",
    "ru": "UD_Russian",
    "la": "UD_Latin",
    "tr": "UD_Turkish",
    "hr": "UD_Croatian",
    "uk": "UD_Ukrainian",
    "hi": "UD_Hindi",
    "sk": "UD_Slovak",
    "sl": "UD_Slovenian",
    "cs": "UD_Czech",
    "kk": "UD_Kazakh",
    "ga": "UD_Irish",
    "de": "UD_German",
    "lv": "UD_Latvian",
    "co": "UD_Coptic",
    "pt": "UD_Portuguese",
    "ca": "UD_Catalan",
    "no": "UD_Norwegian",
    "nl": "UD_Dutch",
    "he": "UD_Hebrew",
    "da": "UD_Danish",
    "fr": "UD_French",
    "pl": "UD_Polish",
    "zh": "UD_Chinese",
    "fa": "UD_Persian",
    "ta": "UD_Tamil",
    "hu": "UD_Hungarian",
    "ja": "UD_Japanese",
    "et": "UD_Estonian",
    "go": "UD_Gothic",
    "eu": "UD_Basque",
    "en": "UD_English",
    "it": "UD_Italian",
    "gl": "UD_Galician",
    "vi": "UD_Vietnamese",
    "ro": "UD_Romanian",
    "el": "UD_Greek",
    "es": "UD_Spanish",
    "bg": "UD_Bulgarian",
    "sa": "UD_Sanskrit",
    "sv": "UD_Swedish",
    "ar": "UD_Arabic",
    "fi": "UD_Finnish",
}

for language_code in languages:
    if language_code == "cs":
        continue
    polyglot_path = f"./results/polyglot/{language_code}/polyglot/vocab_embedding.txt"
    ud_path =  f"./datasets/universal_dependencies/ud-treebanks-v1.4/{language_code_to_folder[language_code]}/vocab.txt"

    with open(polyglot_path) as polyglot_file, open(ud_path) as ud_file:
        polyglot_vocab = set(l.split()[0] for l in polyglot_file)
        ud_vocab = [l.strip() for l in ud_file]
        
        oov = sum(w in polyglot_vocab for w in ud_vocab) / len(ud_vocab)
        print(language_code, oov)

# for language_code in languages:
#     if language_code == "cs":
#         continue
#     print(language_code, language_code_to_folder[language_code][3:])