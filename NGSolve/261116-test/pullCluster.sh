#!/bin/bash
ORIGIN="duvan.henao@172.16.105.104:~/gels/261116-test"
DESTINATION=".."
rsync -avP $ORIGIN $DESTINATION
