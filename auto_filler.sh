#!/usr/bin/expect
spawn apt install nvidia-cudnn
expect "Do you want to continue? [Y/n]"
send "y\r"
expect "1. I decline 2. I Agree"
send "2\r"
interact