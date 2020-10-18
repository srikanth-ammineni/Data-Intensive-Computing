#!/usr/bin/python3
import sys
import collections

prev_id=None
curr_id=None
prev_row=[]

#print the header
header=["Employee ID","Name","Salary" ,"Country","Passcode"]
print(",".join(header))


#Read each line from mapper output
for line in sys.stdin:
    line=line.strip()
    columns=line.split(",") 
    curr_id=columns[0]

    #Add employee name to records which have Null and write to stream
    if curr_id==prev_id and prev_id is not None:
        if len(prevrow)==2:
            columns[1]=prevrow[1]
            print(",".join(columns))
        else:
            prevrow[1]=columns[1]
            print(",".join(prevrow))
    prev_id=curr_id
    prevrow=columns
