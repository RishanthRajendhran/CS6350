echo "Neural Networks"
echo "Building packages..."
pip3 install -e ../../
echo "Building a neural network with gaussian initialization in debug mode"
python3 predict.py -ds bankNote -lrScheme scheme1 -wInit gaussian -debug -T 100 -r 0.1 -width 10 -summary
echo "Building a neural network with zeros initialization in debug mode"
python3 predict.py -ds bankNote -lrScheme scheme1 -wInit zeros -debug -T 100 -r 0.1 -width 5 -summary
echo "Building a neural network with tensorflow"
python3 predict.py -ds bankNote -tensorflow