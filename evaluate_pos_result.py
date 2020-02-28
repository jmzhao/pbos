import os

directory = "./results/polyglot"

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

methods = ["polyglot", "bos", "pbos"]

for language_code in languages:
    if language_code == "cs":
        continue
    print(f"{language_code}", end="\t")
    for method in methods:
        ud_log_file_path = f"{directory}/{language_code}/{method}/ud-log/log.txt"
        with open(ud_log_file_path) as log_file:
            lines = log_file.readlines()[-50:]
            for line in lines:
                if "POS Test Accuracy" in line:
                    print(line[19:-1], end="\t")
    for method in methods:
        ud_log_file_path = f"{directory}/{language_code}/{method}/ud-log/log.txt"
        with open(ud_log_file_path) as log_file:
            lines = log_file.readlines()[-30:]
            macro_f1 = lines[-2].split(",")[1].split(" ")[1]
            print(f"{macro_f1}", end="\t")
    print()
