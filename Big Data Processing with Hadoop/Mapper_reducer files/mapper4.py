#!/usr/bin/python3

import sys

prev_file_name=None


#read each line from mapper
for line in sys.stdin:    
        line=line.strip() 

        #if the record is from join2 file, then insert a column <Name> and make it's value as Null and pass it on to reducer
        if line:
            columns = line.split(",")
            if columns[0]!="Employee ID":
                if len(columns)==2:
                    print(','.join(columns))
                else:
                    columns.insert(1,"null")
                    print(','.join(columns))

    
