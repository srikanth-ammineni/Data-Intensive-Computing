#!/usr/bin/python3
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import re

keywords = ["science" ,"fire", "sea"]


nltk.download('punkt')

prev_words=[]

#read each line from input
for line in sys.stdin:
	line=line.strip()

	#if line is empty, don't do anything
	if not line:
		continue

	#strip any special characters using regex
	line=re.sub(r"[^a-zA-Z0-9]+",' ',line.lower())

	#Tokenize 
	words=line.split()
	if prev_words:
		temp=prev_words
		temp.extend(words)
		line=" ".join(temp)
	else:
		line=" ".join(words)

	#Add words from previous line which could possibly form trigrams
	prev_words=[]	
	if len(words)>=2:
		prev_words.append(words[-2])
		prev_words.append(words[-1])
	elif len(words)==1:
		prev_words.append(words[0])

	#Find trigrams using nltk package
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(line)	
	trigrams=ngrams(tokens,3)
	trigrams=list(trigrams)


	#Replace each keyword with "$" and write to stream as <trigram,1>
	if trigrams:
		for word in keywords:
			for gram in trigrams:
				gram=list(gram)
				for i in range(3):
					if gram[i]==word:
						gram[i]="$"
						gram='_'.join(gram)
						print(gram,1)
						break
