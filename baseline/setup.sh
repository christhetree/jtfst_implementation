#!/bin/sh

wget "https://code.soundsoftware.ac.uk/attachments/download/2696/vamp-plugin-pack-installer-1.0"
chmod +x vamp-plugin-pack-installer-1.0
./vamp-plugin-pack-installer-1.0
wget "https://code.soundsoftware.ac.uk/attachments/download/2708/sonic-annotator-1.6-linux64-static.tar.gz"
tar -xvzf sonic-annotator-1.6-linux64-static.tar.gz
cp sonic-annotator-1.6-linux64-static/sonic-annotator .
rm -r sonic-annotator-1.6-linux64-static
chmod +x sonic-annotator
./sonic-annotator -t pyin_params.n3 example/example.wav -w lab --lab-force
if [ -e example/*.lab ]
then
    echo "Pitch extraction plugin test successful"
else
    echo "Error: Pitch extraction plugin installation unsuccessful"
fi

