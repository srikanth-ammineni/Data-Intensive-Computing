#!/usr/bin/python3
import sys
import re

#read each line from input
for line in sys.stdin:  
    line=line.strip()

    # strip any special characters using regex
    line=re.sub(r"[^a-zA-Z0-9]+",' ',line.lower())

    #Tokenize each line
    words=line.split()

    #Write to stream <word,1>
    for word in words:
        print(word,1)
