#!/usr/bin/python3
import sys

prev_word=None
curr_word=None
docids=[]

#read each line from mapper output
for line in sys.stdin:
    line=line.strip()
    if line:
        word,count=line.split()
        docid=int(count)
        curr_word=word

        #Accumulate all the documentids and write to stream <word,docids>
        if curr_word!=prev_word and prev_word is not None:
            print(prev_word+": "+str(docids))
            docids=[]
        if docid not in docids:
            docids.append(docid)
        prev_word=curr_word

#Write to stream the final word
print(curr_word+": "+str(docids))
