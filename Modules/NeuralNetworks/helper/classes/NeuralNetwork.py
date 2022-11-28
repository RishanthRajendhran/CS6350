from Modules.NeuralNetworks.helper.imports.packageImports import np, plt
import Modules.Results

class NeuralNetwork:
    def __init__(self, layers=[], loss="lms"):
        self.weights = []
        self.numParams = 0
        self.loss = loss
        self.layers = []
        if len(layers):
            for i in range(len(layers)):
                self.add(layers[i])

    def __call__(self, X, return_activations=False):
        if len(self.layers) == 0:
            print("Add layers to the Neural Network first!")
            return 
        if return_activations:
            allActivations = {
                "inputLayer": {
                    "input": X
                }
            }
        activation = np.array(X).copy()
        for layer in self.layers:
            if return_activations:
                allActivations[layer._getName()] = {
                    "input": activation
                }
            activation = layer(activation)
            if return_activations:
                allActivations[layer._getName()]["activations"] = activation
        if return_activations:
            return activation, allActivations
        else:
            return activation

    def getLoss(self, X, Y):
        if self.loss == "lms":
            preds = self(X)
            preds = np.ndarray.flatten(preds)
            return np.sum((preds-Y)**2/2)

    def backPropagate(self, Y, allActivations):
        grads = {}
        preds = allActivations[self.layers[-1]._getName()]["activations"]
        curGrad = {}
        if self.loss == "lms":
            loss_z = np.array(preds-Y)
            curGrad["loss_z"] = np.zeros((len(loss_z), 1+loss_z.shape[1]))
            curGrad["loss_z"][:,1:] = loss_z.copy()
        else: 
            print("Unrecognized losss function!")
            return None
        for i in range(len(self.layers)-1,-1,-1):
            X = np.array(allActivations[self.layers[i]._getName()]["input"])
            curGrad, prevGrad = self.layers[i]._backPropagate(X, curGrad)
            grads[self.layers[i]._getName()] = curGrad
            curGrad = prevGrad
        return grads

    def train(self, X, Y, T=100, lr=0.01, lrScheme="constant", hyperparameters={}, debug=False, savePlot = False):
        learningRate = lr

        allLosses = []
        for t in range(T):
            newOrder = np.random.permutation(len(X))
            for i in newOrder:
                x = X[i,:].copy()
                y = Y[i].copy()
                if lrScheme == "scheme1":
                    learningRate = lr/(1+((lr/hyperparameters["a"])*t))
                elif lrScheme == "scheme2":
                    learningRate = lr/(1+t)
                pred, allActivations = self([x], return_activations=True)
                grads = self.backPropagate([y], allActivations)
                for i in range(len(self.layers)):
                    oldW = self.layers[i]._getWeights()
                    gradW = grads[self.layers[i]._getName()]["loss_w"]
                    self.weights[i] = self.layers[i]._updateWeights(oldW - learningRate*gradW)
            loss = self.getLoss(X, Y)
            allLosses.append(loss)
            if debug and t%10==0:
                print(f"\t\t\tIteration {t+1}: Loss={loss}")
        allLosses = np.array(allLosses)
        
        if debug and T%10==0:
            loss = self.getLoss(X, Y)
            print(f"\t\t\tEnd of training: Loss={loss}")

        if debug or savePlot:
            plt.plot(allLosses)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            if savePlot:
                hyps =  ""
                for key in hyperparameters.keys():
                    hyps += "_"
                    hyps += str(hyperparameters[key])
                plt.savefig(f"{str(list(Modules.Results.__path__)[0])}/NeuralNetworkLoss_{T}_{lr}_{lrScheme}_{hyps}.png")
                plt.clf()
            else:
                plt.show()

    def add(self, layer):
        layerInputShape = layer._getInputShape()
        layerName = layer._getName()
        if layerName == None:
            layer._updateName(len(self.layers))
        else: 
            for l in self.layers:
                if l._getName() == layerName:
                    layer._updateName(len(self.layers), layerName)
        if layerInputShape == None:
            if len(self.layers) > 0:
                layer._updateInputShape((self.layers[-1]._getNumNeurons(),))
        elif len(self.layers) > 0 and layerInputShape != (self.layers[-1]._getNumNeurons(),):
            layer._updateInputShape((self.layers[-1]._getNumNeurons(),))
        layerWeights = layer._getWeights()
        if (len(self.layers)==0 or len(self.weights)) and len(layerWeights):
            self.weights.append(layerWeights)
            self.numParams += layer._getNumParams()

        self.layers.append(layer)
    
    def summary(self):
        if len(self.weights) == 0:
            print("Model not yet built!")
            return 
        print("--------------------------------------------")
        print("Summary")
        print("Neural Network")
        print(f"Number of layers: {len(self.layers)}")
        for i in range(len(self.layers)):
            print(f"Layer {i+1}:")
            self.layers[i]._printLayer()
            print("+++++++++++")
        print(f"Total number of parameters: {self.numParams}")
        print("--------------------------------------------")