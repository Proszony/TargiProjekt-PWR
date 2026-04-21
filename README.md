Projekt Zespolowy
================

Skryp w bashu do uruchamiania streamingu na raspberry PI obrazu z kamery przez sieć na podane ip i port.
Do przesyłu wykorzystano bibliotekę ffmpeg, którą należy wcześniej pobrać na malinkę.

#!/bin/bash
ip=$1
port=$2

ffmpeg -f v4l2 -framerate 15 -video_size 640x480 -i /dev/video0 \
-c:v h264_v4l2m2m -b:v 1M -f mpegts udp://$ip:$port
