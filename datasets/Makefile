TARGET = 20k.txt pg16457.txt

all: $(TARGET)

20k.txt:
	wget https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt

pg16457.txt:
	wget http://www.gutenberg.org/cache/epub/16457/pg16457.txt

clean:
	rm -f $(TARGET)