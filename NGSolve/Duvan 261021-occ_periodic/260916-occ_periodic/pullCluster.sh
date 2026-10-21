#!/bin/bash
ORIGIN="phernandez@172.16.105.104:~/260916-occ_periodic"
DESTINATION="."
rsync -avP $ORIGIN $DESTINATION
