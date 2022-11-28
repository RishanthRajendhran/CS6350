from Modules.NeuralNetworks.helper.imports.packageImports import np

class Dense:
    def __init__(self, numNeurons, activation="linear", weightsInitialization="gaussian", inputShape=None, name=None, weights=[]):
        self.numNeurons = numNeurons
        self.inputShape = inputShape
        self.activation = activation
        self.weightsInitialization = weightsInitialization
        self.name = name
        self.weights = np.array(weights)
        if len(self.weights) != 0:
            if self.inputShape == None:
                self.inputShape = (self.weights.shape[1]-1,)
            elif self.inputShape != (self.weights.shape[1]-1,):
                print("Mismatch!")
                weights = []
        if self.inputShape != None and len(weights) == 0:
            self.weights = np.ones((self.numNeurons, 1+self.inputShape[0]))
            if self.weightsInitialization == "gaussian":
                self.weights[:, 1:] = np.random.normal(size=(self.numNeurons, self.inputShape[0]))
            elif self.weightsInitialization == "zeros":
                self.weights[:, 1:] = np.zeros((self.numNeurons, self.inputShape[0]))

    def __call__(self, X):
        newX = np.ones((X.shape[0],X.shape[1]+1))
        newX[:,1:] = X.copy()

        #Aggregation
        result = np.dot(newX,self.weights.T)

        #Activation
        if self.activation == "sigmoid":
            result = 1/(1+np.exp(-result))

        return result

    def _getDerivativeOfActivation(self, input):
        if self.activation == "linear":
            return np.ones(self.weights.shape)[np.newaxis,:]
        elif self.activation == "sigmoid":
            z = self(input)
            z = np.ndarray.flatten(z)
            z = z*(1-z)
            out = []
            for i in range(len(z)):
                curOut = []
                for j in range(self.weights[i].shape[0]):
                    curOut.append(z[i])
                out.append(curOut)
            return np.array(out)[np.newaxis,:]
            
        return None

    #nextGrad.keys() = ["weights", "WRTactivation", "WRTinput", "WRTweights"]
    #Make sure input is without augmentation
    def _backPropagate(self, X, curGrad):
        augX = np.ones((X.shape[0], X.shape[1]+1))
        augX[:, 1:] = X.copy()

        d = self._getDerivativeOfActivation(X)

        # print("loss_z: ", curGrad["loss_z"][:,1:].T)
        # print(f"d: {d}")
        # print(f"augX: {augX}")
        # print(f"d*augX: {d*augX}")
        curGrad["loss_w"] = np.sum(curGrad["loss_z"][:,1:].T*((d*augX)), axis=0)
        # print(curGrad["loss_w"])

        w = self.weights
        prevGrad = {}
        prevGrad["loss_z"] = np.sum(curGrad["loss_z"][:,1:].T*(d*w), axis=1)

        return curGrad, prevGrad
        


    def _updateInputShape(self, inputShape):
        if self.inputShape != None and len(self.weights) != 0 and self.weights.shape != (self.numNeurons, 1+inputShape[0]):  
            print(f"Ignoring input weights to layer {self.name} of wrong shape.")
        self.inputShape = inputShape
        self.weights = np.ones((self.numNeurons, 1+self.inputShape[0]))
        if self.weightsInitialization == "gaussian":
            self.weights[:, 1:] = np.random.normal(size=(self.numNeurons, self.inputShape[0]))
        elif self.weightsInitialization == "zeros":
            self.weights[:, 1:] = np.zeros((self.numNeurons, self.inputShape[0]))
    
    def _updateName(self, layerNum, name="dense"):
        self.name = f"{name}_{layerNum}"

    def _updateWeights(self, newW):
        if self.weights.shape != newW.shape:
            print("Wrong shape of weights!")
            return
        self.weights = newW
        return self.weights

    def _getWInit(self):
        return self.weightsInitialization

    def _getInputShape(self):
        return self.inputShape

    def _getNumNeurons(self):
        return self.numNeurons

    def _getWeights(self):
        return self.weights

    def _getNumParams(self):
        if self.inputShape == None:
            return None
        return (1+self.inputShape[0])*self.numNeurons

    def _getName(self):
        return self.name

    def _printLayer(self):
        print(f"\t{self.name} (Dense)")
        print(f"\tInput shape: {self.inputShape}")
        print(f"\tNumber of neurons: {self.numNeurons}")
        print(f"\tActivation: {self.activation}")
        print(f"\tNumber of parameters: {self._getNumParams()}")
        print(f"Weights:\n{self.weights}")