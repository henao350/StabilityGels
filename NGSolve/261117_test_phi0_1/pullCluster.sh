#!/bin/bash
ORIGIN="duvan.henao@172.16.105.104:~/gels/261117_test_phi0_1"
DESTINATION=".."
rsync -avP $ORIGIN $DESTINATION
