#!/usr/bin/python3
import sys
import collections

prev_word=None
curr_word=None
word_count=0


Hashset=collections.defaultdict()
for line in sys.stdin:
    line=line.strip()
    word,count=line.split()
    count=int(count)
    curr_word=word
    if curr_word!=prev_word and prev_word is not None:
        Hashset[prev_word]=word_count
        #print(prev_word,word_count)
        word_count=0
    word_count+=count
    prev_word=curr_word
Hashset[curr_word]=word_count
#print(curr_word,word_count)
top10=sorted(Hashset.items(),key=lambda x:-x[1])[:10]
for i,j in top10:
	print(i+","+str(j))