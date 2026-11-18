#!/bin/bash
ORIGIN="../261118-test2-dry"
DESTINATION="duvan.henao@172.16.105.104:~/gels/"
rsync -avP $ORIGIN $DESTINATION
