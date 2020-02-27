.PHONY: all
all: 20k.txt unigram_freq.csv GoogleNews-vectors-negative300.pkl

# A list of the 20,000 most common English words
# See https://github.com/first20hours/google-10000-english
20k.txt:
	wget https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt

# 1/3 Million Most Frequent English Words on the Web
# See https://www.kaggle.com/rtatman/english-word-frequency

unigram_freq.csv:
	wget https://raw.githubusercontent.com/jai-dewani/Word-completion/master/unigram_freq.csv
	sed -i 's/,/\t/g' unigram_freq.csv
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