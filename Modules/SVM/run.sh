echo("SVM")
echo("Running primal form of SVM with scheme1 learning rate decomposition")
python3 predict.py -ds bankNote -form primal -lrScheme scheme1 -r 0.000001 -C 0.572737686139748
echo("Running primal form of SVM with scheme2 learning rate decomposition")
python3 predict.py -ds bankNote -form primal -lrScheme scheme2 -r 0.00001 -C 0.572737686139748
echo("Running dual form of SVM with scheme1 learning rate decomposition")
python3 predict.py -ds bankNote -form dual -lrScheme scheme1 -C 0.572737686139748
echo("Running dual form of SVM (gaussian kernel) with scheme1 learning rate decomposition")
python3 predict.py -ds bankNote -form dual -lrScheme scheme2 -C 0.8018327605956472 -gamma 0.1 -kernel gaussian

