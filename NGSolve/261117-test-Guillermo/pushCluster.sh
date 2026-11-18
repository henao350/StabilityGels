#!/bin/bash
ORIGIN="../261117-test-Guillermo"
DESTINATION="duvan.henao@172.16.105.104:~/gels/"
rsync -avP $ORIGIN $DESTINATION
