TARGET = 20k.txt pg16457.txt unigram_freq.csv

all: $(TARGET)

20k.txt:
	wget https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt

pg16457.txt:
	wget http://www.gutenberg.org/cache/epub/16457/pg16457.txt

unigram_freq.csv:
	wget -O unigram_freq.zip "https://storage.googleapis.com/kaggle-data-sets/2367/3976/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1581943782&Signature=oU9Qa1fgHq36iDOv81tnFCqiFZ79fa%2F2Wc0gmtFM5lvEF57cYNfao4GzFbQ4zhS9yzioiOEPwgIeEoa0w7gvQxaBiYE0HbFGXajUll1ahntsXEl7x485Jn0f%2FPuquNJlUgYd7a1szewL6m5sdXFXSNJSNErhMvjNfce3HgU0hdCtRAMAebESa3UJIv1BBZQhHjMRLshh%2B9oO8ahnpwouDmJdIpxaV6oBZWnexUXVEIeDn55g77O5QLMF2uDDsbKLkPkiO7gmZ3ZxfBAtE8LRpWVNzw0syyWRUroswRiJREdXBuMUlDY9ZRyXah09kICLcMPEjsaLBfuEuRbc1NIzZw%3D%3D&response-content-disposition=attachment%3B+filename%3Denglish-word-frequency.zip"
	unzip unigram_freq.zip
	rm -f unigram_freq.zip
	tail -n +2 unigram_freq.csv > unigram_freq.csv.tmp
	mv unigram_freq.csv.tmp unigram_freq.csv

clean:
	rm -f $(TARGET)