#!/bin/bash
ORIGIN="../261116-test"
DESTINATION="duvan.henao@172.16.105.104:~/gels/"
rsync -avP $ORIGIN $DESTINATION
