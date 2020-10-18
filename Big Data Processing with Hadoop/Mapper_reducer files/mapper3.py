#!/usr/bin/python3
import fileinput
import re
import sys
import os

#Assign doc id's for each input file
docids={"arthur.txt":1,"james.txt":2,"leonardo.txt":3}

#Read each line from input
for line in sys.stdin:
	if line:
	    line=line.strip()
	    #strip any speciai characters using regex
	    line=re.sub(r"[^a-zA-Z0-9]+",' ',line.lower())

	 	#tokenize 
	    words=line.split()

	    #find the file name that is currently being read
	    filepath=os.environ.get('map_input_file','stdin')
	    filename=os.path.split(filepath)[-1]

	    #Write to stream <word,docid>
	    for word in words:
	    	print(word,docids[filename])
