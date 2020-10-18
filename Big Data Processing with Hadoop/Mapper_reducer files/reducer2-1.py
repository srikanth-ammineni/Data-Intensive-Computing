#!/usr/bin/python3
import sys
import collections

prev_word=None
curr_word=None
word_count=0

Hashmap=collections.defaultdict()

#read each line from mapper output
for line in sys.stdin:
    line=line.strip()
    word,count=line.split()
    count=int(count)
    curr_word=word

    #aggregate the counts and add to hashmap
    if curr_word!=prev_word and prev_word is not None:
        Hashmap[prev_word]=word_count
        word_count=0
    word_count+=count
    prev_word=curr_word
Hashmap[curr_word]=word_count

#find the local top 10 most occuring trigrams
top10=sorted(Hashmap.items(),key=lambda x:-x[1])[:10]

#Write to stream
for i,j in top10:
	print(i+","+str(j))