echo "Building packages..."
pip3 install -e ../../
echo "Car data set:"
python3 predict.py car
echo "--------------"
echo "Bank Data set:"
python3 predict.py bank 