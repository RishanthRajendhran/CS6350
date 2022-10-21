echo "Building packages..."
pip3 install -e ../../
echo "Running Linear Regression (with Batch GD) over concrete dataset with a learning rate of 0.00001 in debug mode"
python3 predict.py -ds concrete -debug -lr 0.00048828
echo "Running Linear Regression (with Stochastic GD) over concrete dataset with a learning rate of 0.00001 in debug mode"
python3 predict.py -ds concrete -debug -gd stochastic -lr 0.00048828