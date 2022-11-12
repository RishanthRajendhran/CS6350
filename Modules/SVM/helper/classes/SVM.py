from Modules.helper.imports.packageImports import np
from Modules.SVM.helper.imports.packageImports import minimize
from Modules.SVM.helper.functions.performSGD import performSGD
from Modules.SVM.helper.functions.updateWeightsSVM import updateWeightsSVM
from Modules.SVM.helper.functions.getObjectiveSVM import getObjectiveSVM

class SVM:
    def __init__(self,form="primal",kernel=None):
        self.weights = None
        self.alphas = None
        self.K = None
        self.kernel = kernel
        self.X = None
        self.Y = None
        self.form = form
        self.gamma = 0.0075

    def fit(self, X, Y, T=100, lr=0.01, lrScheme="constant", hyperparameters={}, debug=False, savePlot=False, weights=None):
        if self.form == "primal":
            if weights == None or len(weights) != len(X)+1:
                weights = np.zeros((X.shape[1]+1,))
            
            newX = np.ones((X.shape[0], X.shape[1]+1))
            newX[:,:-1] = X.copy()
            
            if hyperparameters.get("C") == None:
                hyperparameters["C"] = 1
                print("SVM: C not specified, setting C = " + str(hyperparameters["C"]))

            if hyperparameters.get("N") == None:
                hyperparameters["N"] = len(X)
                print(f"SVM: N not specified, setting N = " + str(hyperparameters["N"]))

            weights = performSGD(newX, Y, updateWeightsSVM, getObjectiveSVM,  T, lr, lrScheme, hyperparameters, debug, savePlot, weights)

            self.weights = weights
            C = hyperparameters["C"]
            np.save(f"weights_{lr}_{C}", weights)
        elif self.form == "dual":
            gamma = hyperparameters["gamma"]
            C = hyperparameters["C"]

            newX = np.ones((X.shape[0], X.shape[1]+1))
            newX[:,:-1] = X.copy()

            N = len(Y)
            constraints = [
                # sum_i (alpha_i y_i) = 0
                {'type': 'eq',   
                'fun': lambda alpha: np.dot(alpha, (2*Y-1))
                },
                #Non-negativity constraints
                {'type': 'ineq',    
                'fun': lambda alpha: np.dot(np.eye(N), alpha)
                },
                #alpha_i <= C rewritten as C - alpha_i >= 0
                {'type': 'ineq',    
                'fun': lambda alpha: (hyperparameters["C"] * np.ones(N)) - np.dot(np.eye(N), alpha)
                },
            ] 
            
            YYT = np.matmul((2*Y[:,np.newaxis]-1),(2*Y[:,np.newaxis]-1).T)
            XXT = np.matmul(newX,newX.T)
            if self.kernel == "gaussian":
                if hyperparameters.get("gamma") != None:
                    self.gamma = hyperparameters["gamma"]
                self.K = lambda x, y: np.exp(-(np.linalg.norm(x[:, np.newaxis] - y,axis=2)**2)/self.gamma)
                XXT = self.K(newX, newX)
            dualObj = lambda alpha:  (1/2)*np.dot(alpha,np.dot(alpha, YYT * XXT)) - np.sum(alpha)

            res = minimize(dualObj, 
                np.ones(len(newX)), 
                method='SLSQP', 
                constraints=constraints)
            
            self.alphas = res.x.copy()
            if self.kernel == "gaussian":
                np.save(f"kernelAlphas_{C}_{gamma}.npy",self.alphas)
            else:
                np.save(f"alphas_{C}.npy",self.alphas)

            supportVectors = np.where(np.round(self.alphas, 6) != 0)[0]
            
            if self.kernel == "gaussian":
                np.save(f"kernelSupportVectors_{lr}_{C}_{gamma}", supportVectors)
            else:
                np.save(f"supportVectors_{lr}_{C}", supportVectors)

            print(f"\t\t\tNo. of support vectors: {len(supportVectors)}")

            # self.K = lambda x, y: np.exp(-(np.linalg.norm(x[:, np.newaxis] - y,axis=2)**2)/self.gamma)
            # self.alphas = np.load(f"alphas.npy")
            
            self.X = newX.copy()
            self.Y = Y.copy()

            weights = np.sum(self.alphas*self.X.T*(2*self.Y-1),axis=1)
            np.save(f"kernelWeights_{lr}_{C}_{gamma}", weights)

    def predict(self, X):
        if self.form == "primal":
            newX = np.ones((X.shape[0], X.shape[1]+1))
            newX[:,:-1] = X.copy()
            return (np.sign(np.dot(newX, self.weights))+1)/2
        elif self.form == "dual":
            if self.kernel == "gaussian":
                newX = np.ones((X.shape[0], X.shape[1]+1))
                newX[:,:-1] = X.copy()
                preds = np.sum((self.alphas*(2*self.Y-1))[:,np.newaxis]*self.K(self.X, newX),axis=0)
                return (np.sign(preds)+1)/2
            else:
                newX = np.ones((X.shape[0], X.shape[1]+1))
                newX[:,:-1] = X.copy()
                preds = np.sum((self.alphas*(2*self.Y-1))[:,np.newaxis]*np.dot(self.X, newX.T),axis=0)
                return (np.sign(preds)+1)/2
