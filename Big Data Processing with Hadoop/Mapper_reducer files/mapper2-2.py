#!/usr/bin/python3
import sys

#read each line from output of mapper1 and pass it on to reducer 
for line in sys.stdin:
	line=line.strip()
	if line:
		print(line)
