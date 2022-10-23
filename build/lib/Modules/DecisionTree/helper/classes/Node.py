#Class Name         : Node
#Description        : Class definition for a node in a decision tree for a binary classification task
#Data members       : 
#                     attr              : Attribute on which split is done
#                     children          : List of all child nodes
#                     childrenAttrVal   :   List of value of attribute on which every child is split
#                     depth             : Depth of the node in the decision tree
#                     isLeafNode        : Boolean variable indicating whether the node is a leaf in the 
#                                         decision tree
#                                         Default: False
#                     label             : Majority label of data samples at the node
#Member functions   :
#                    addChild   : Add a child node 
#                    printTree  : Print node and its children
#                    predict    : Predict target label for a data instance
class Node:
    def __init__(self,attr,depth,isLeafNode=False,label=None):
        self.attr = attr 
        self.children = []
        self.childrenAttrVal = []
        self.depth = depth
        self.isLeafNode = isLeafNode
        self.label = label
    def addChild(self, child, attrVal):
        self.children.append(child)
        self.childrenAttrVal.append(attrVal)
    def printTree(self,n=0):
        st = "\t"*n
        print(f"{st}Attr {self.attr}, Depth {self.depth}")
        if self.isLeafNode:
            print(f"{st}Label: {self.label}")   
        for ch in range(len(self.children)):
            print(f"{st}\tAttrVal = {self.childrenAttrVal[ch]}")
            self.children[ch].printTree(n+2)
    def predict(self, x):
        if self.isLeafNode:
            return self.label
        return self.children[self.childrenAttrVal.index(x[self.attr])].predict(x)