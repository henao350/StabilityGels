#!/bin/bash
ORIGIN="duvan.henao@172.16.105.104:~/gels/261118-test3-phi0_0_9/"
DESTINATION=".."
rsync -avP $ORIGIN $DESTINATION
