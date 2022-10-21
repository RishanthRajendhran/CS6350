echo "Building packages..."
pip3 install -e ../../../
echo "Running AdaBoost over bank dataset with at most 500 decision stumps in debug mode"
python3 predict.py -ds bank -nwl 500 -tune -sr -debug