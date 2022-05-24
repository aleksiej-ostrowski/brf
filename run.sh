bash brf.sh
./brf.o --selftest 0
cd ./compare
python3 predict.py
cd ..
