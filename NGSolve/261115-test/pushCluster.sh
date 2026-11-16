#!/bin/bash
ORIGIN="../260916-occ_periodic"
DESTINATION="phernandez@172.16.105.104:~/"
rsync -avP $ORIGIN $DESTINATION
