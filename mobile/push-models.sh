#!/bin/zsh
# Copy the models onto a connected Android phone (or emulator) for the debuggable test build, so the
# app starts without downloading anything. Needs adb and a build made with KLARA_DEBUGGABLE=1.
set -e
ADB=${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}
PKG=com.jsbeaudry.klaramobile
HERE=${0:A:h}
$ADB shell mkdir -p /data/local/tmp/km
for f in $HERE/models/asr/*.gguf; do $ADB push "$(readlink -f $f)" /data/local/tmp/km/; done
$ADB push $HERE/models/pocket /data/local/tmp/km/
$ADB shell chmod -R a+rX /data/local/tmp/km
$ADB shell "run-as $PKG sh -c 'mkdir -p files/models/asr files/models/pocket && cp /data/local/tmp/km/*.gguf files/models/asr/ && cp -r /data/local/tmp/km/pocket/. files/models/pocket/'"
$ADB shell rm -rf /data/local/tmp/km
echo "models copied; open Klara on the phone"
