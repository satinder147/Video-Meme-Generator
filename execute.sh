mkdir temp
TF_CPP_MIN_LOG_LEVEL=3 python3 -W ignore get_face.py $1 $2 #tensorflow flag for minimum logging
echo "face generation completed"
python3 -W ignore run.py $3 $4 $5 $6
echo "deleting temporary files"
ffmpeg -i $4 -ab 160k -ac 2 -ar 44100 -vn temp/audio.wav
ffmpeg -y -i temp/audio.wav  -r 30 -i $6  -filter:a aresample=async=1 -c:a flac -c:v copy result.mkv
rm $6
rm -r temp
echo "completed"
