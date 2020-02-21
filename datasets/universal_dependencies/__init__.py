import os
import subprocess as sp
import tarfile

dir_path = os.path.dirname(os.path.realpath(__file__))

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
    "fi": "UD_Finnish"
}


def get_universal_dependencies_path(language_code):
    tgz_path = f"{dir_path}/ud-treebanks-v1.4.tgz"
    language_folder_path = f"{dir_path}/ud-treebanks-v1.4/{language_code_to_folder[language_code]}"
    vocab_path = f"{language_folder_path}/vocab.txt"
    data_path = f"{language_folder_path}/combined.pkl"

    if not os.path.exists(tgz_path):
        url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1827/ud-treebanks-v1.4.tgz?sequence=4&isAllowed=y"
        sp.run(f"wget -O {tgz_path} {url}".split())

    if not os.path.exists(f"{dir_path}/ud-treebanks-v1.4"):
        with tarfile.open(tgz_path) as tar:
            tar.extractall(dir_path)

    if not os.path.exists(vocab_path) or not os.path.exists(data_path):
        sp.run(f"""
            python make_dataset.py \
              --training-data {language_folder_path}/{language_code}-ud-train.conllu \
              --dev-data {language_folder_path}/{language_code}-ud-dev.conllu \
              --test-data {language_folder_path}/{language_code}-ud-test.conllu \
              --output {data_path} \
              --vocab {vocab_path}
            """.split())

    return data_path, vocab_path


if __name__ == '__main__':
    languages = ['kk', 'ta', 'lv', 'vi', 'hu', 'tr', 'el', 'bg', 'sv', 'eu', 'ru', 'da', 'id', 'zh', 'fa', 'he', 'ro',
                 'en', 'ar', 'hi', 'it', 'es', 'cs']
    for language_code in languages:
        print(get_universal_dependencies_path(language_code))

