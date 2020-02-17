.PHONY: all
all: 20k.txt unigram_freq.csv GoogleNews-vectors-negative300.pkl

# A list of the 20,000 most common English words
# See https://github.com/first20hours/google-10000-english
20k.txt:
	wget https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt

# 1/3 Million Most Frequent English Words on the Web
# See https://www.kaggle.com/rtatman/english-word-frequency
unigram_freq.zip:
	wget -O unigram_freq.zip "https://storage.googleapis.com/kaggle-data-sets/2367/3976/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1581943782&Signature=oU9Qa1fgHq36iDOv81tnFCqiFZ79fa%2F2Wc0gmtFM5lvEF57cYNfao4GzFbQ4zhS9yzioiOEPwgIeEoa0w7gvQxaBiYE0HbFGXajUll1ahntsXEl7x485Jn0f%2FPuquNJlUgYd7a1szewL6m5sdXFXSNJSNErhMvjNfce3HgU0hdCtRAMAebESa3UJIv1BBZQhHjMRLshh%2B9oO8ahnpwouDmJdIpxaV6oBZWnexUXVEIeDn55g77O5QLMF2uDDsbKLkPkiO7gmZ3ZxfBAtE8LRpWVNzw0syyWRUroswRiJREdXBuMUlDY9ZRyXah09kICLcMPEjsaLBfuEuRbc1NIzZw%3D%3D&response-content-disposition=attachment%3B+filename%3Denglish-word-frequency.zip"

unigram_freq.csv: unigram_freq.zip
	unzip unigram_freq.zip
	tail -n +2 unigram_freq.csv > unigram_freq.csv.tmp
	mv unigram_freq.csv.tmp unigram_freq.csv

# word2vec pre-trained Google News corpus (3 billion running words) word vector model
# 3 million 300-dimension English word vectors
# See https://github.com/mmihaltz/word2vec-GoogleNews-vectors
GoogleNews-vectors-negative300.bin:
	wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
	gunzip GoogleNews-vectors-negative300.bin.gz

GoogleNews-vectors-negative300.pkl: GoogleNews-vectors-negative300.bin
	python ./converter/word2vec.py --input "GoogleNews-vectors-negative300.bin" --output "GoogleNews-vectors-negative300.pkl"


.PHONY: clean
clean:
	git clean -dfX .