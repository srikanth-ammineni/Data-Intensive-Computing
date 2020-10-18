#!/usr/bin/python3
import sys

prev_word=None
curr_word=None
word_count=0


#Read each line from output of the mapper
for line in sys.stdin:
    line=line.strip()
    word,count=line.split()
    count=int(count)
    curr_word=word

    #Aggregate the counts for each word and print to output
    if curr_word!=prev_word and prev_word is not None:
        print(prev_word,word_count)
        word_count=0
    word_count+=count
    prev_word=curr_word

#print the final word
print(curr_word,word_count)
