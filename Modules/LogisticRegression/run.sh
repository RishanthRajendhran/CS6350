echo "Logistic Regression"
echo "Building packages..."
pip3 install -e ../../
echo "Building a logistic regression model with gaussian prior (MAP)"
python3 predict.py -ds bankNote -obj map -variance 100 -r 0.001 -d 10 
echo "Building a logistic regression model with uniform prior (ML)"
python3 predict.py -ds bankNote -obj ml -r 1e-5 -d 10 