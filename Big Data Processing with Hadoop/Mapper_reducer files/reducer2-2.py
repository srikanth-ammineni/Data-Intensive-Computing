#!/usr/bin/python3
import sys
import collections

hashmap=collections.defaultdict()

#Read each line from mapper output
for line in sys.stdin:
	line=line.strip()
	if line:
		word,count=line.split(",")
		count=int(count)

		#Store top 10 most occuring trigrams at any time
		if len(hashmap)==10:
			min_key=min(hashmap.keys(),key=(lambda k:hashmap[k]))
			min_val=hashmap[min_key]
			if count > min_val:
				del hashmap[min_key]
				hashmap[word]=count
		else:
			hashmap[word]=count


#Write to stream
for k,v in sorted(hashmap.items(),key=lambda x: x[1], reverse=True):
	print(k+": "+str(v))


