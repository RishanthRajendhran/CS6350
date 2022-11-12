echo "Building packages..."
pip3 install -e ../../
echo "Building standard perceptron with learning rate = 0.001 and numEpochs = 10 in debug mode"
python3 predict.py -ds bankNote -debug -pa standard -T 10 -r 0.001
echo "Building voted perceptron with learning rate = 0.1 and numEpochs = 10 in debug mode"
python3 predict.py -ds bankNote -debug -pa voted -T 10 -r 0.1
echo "Building averaged perceptron with learning rate = 0.1 and numEpochs = 10 in debug mode"
python3 predict.py -ds bankNote -debug -pa averaged -T 10 -r 0.1
echo "Building (gaussian) kernel perceptron with learning rate = 0.1 and numEpochs = 10 in debug mode"
python3 predict.py -ds bankNote -debug -pa gaussianKernel -T 10 -r 0.1

