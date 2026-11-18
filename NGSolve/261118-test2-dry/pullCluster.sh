#!/bin/bash
ORIGIN="duvan.henao@172.16.105.104:~/gels/261118-test2-dry"
DESTINATION=".."
rsync -avP $ORIGIN $DESTINATION
