echo "Building packages..."
pip3 install -e ../../../
echo "Running Bagging over bank dataset with at most 500 decision trees in debug mode"
python3 predict.py -ds bank -nl 500 -tune -sr -debug
echo "Growing a Random Forest over bank dataset with at most 500 decision trees choosing 2/4/6 attributes in debug mode"
python3 predict.py -ds bank -nl 500 -nal 2 4 6 -rf -tune -sr -debug