from Modules.helper.imports.packageImports import np
import Modules.Datasets.car
import Modules.Datasets.bank
import Modules.Datasets.creditCard
import Modules.Datasets.concrete

dataSets = {
    "car": {
        "trainPath": str(list(Modules.Datasets.car.__path__)[0])+ "/train.csv",
        "testPath": str(list(Modules.Datasets.car.__path__)[0])+ "/test.csv",
        "attrNames": ["buying","maint","doors","persons","lug_boot","safety","label"],
        "labelNames": ["No", "Yes"],
        "categoricalAttrs": [0,1,2,3,4,5],
        "numericalAttrs": [],
        "attrVals": {
            0:   ["vhigh", "high", "med", "low"],
            1:    ["vhigh", "high", "med", "low"],
            2:    ["2", "3", "4", "5more"],
            3:  ["2", "4", "more"],
            4: ["small", "med", "big"],
            5:   ["low", "med", "high"],
        },
        "labelType": str,
        "maxTreeDepths": np.arange(6)+1,
        "skipHeader": True,
    },
    "bank": {
        "trainPath": str(list(Modules.Datasets.bank.__path__)[0])+ "/train.csv",
        "testPath": str(list(Modules.Datasets.bank.__path__)[0])+ "/test.csv",
        "attrNames": ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"],
        "labelNames": ["no", "yes"],
        "categoricalAttrs": [1,2,3,4,6,7,8,10,15],
        "numericalAttrs": [0,5,9,11,12,13,14],
        "attrVals": {
            1:   ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
            2:    ["married","divorced","single"],
            3:    ["unknown","secondary","primary","tertiary"],
            4:  ["yes","no"],
            6: ["yes","no"],
            7:   ["yes","no"],
            8:   ["unknown","telephone","cellular"],
            10:    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            15:    ["unknown","other","failure","success"],
        },
        "labelType": str,
        "maxTreeDepths": np.arange(16)+1,
        "skipHeader": True,
    },
    "concrete": {
        "trainPath": str(list(Modules.Datasets.concrete.__path__)[0])+ "/train.csv",
        "testPath": str(list(Modules.Datasets.concrete.__path__)[0])+ "/test.csv",
        "attrNames": ["cement","slag","flyAsh","water","sp","coarseAggr","fineAggr"],
        "labelNames": None,
        "categoricalAttrs": [],
        "numericalAttrs": [0,1,2,3,4,5,6],
        "attrVals": None,
        "labelType": float,
        "maxTreeDepths": np.arange(16)+1,
        "skipHeader": False,
    },
    "creditCard": {
        "trainPath": str(list(Modules.Datasets.creditCard.__path__)[0])+ "/trainX.csv",
        "testPath": str(list(Modules.Datasets.creditCard.__path__)[0])+ "/testX.csv",
        "attrNames": ["id","limit_bal","sex","education","marriage","age","pay_0","pay_2","pay_3", "pay_4", "pay_5", "pay_6", "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6", "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"],
        "labelNames": ["0", "1"],
        "categoricalAttrs": [],
        "numericalAttrs": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
        "attrVals": None,
        "labelType": str,
        "maxTreeDepths": np.arange(16)+1,
        "skipHeader": False,
    },
    "iris": {
        #Dummy file Paths
        "trainPath": str(list(Modules.Datasets.creditCard.__path__)[0])+ "/trainX.csv",
        #No test file as of now
        "testPath": str(list(Modules.Datasets.creditCard.__path__)[0])+ "/trainX.csv",
        "labelNames": ["no", "yes"],
        "categoricalAttrs": [],
        "numericalAttrs": [0,1,2,3],
        "attrVals": None,
        "labelType": str,
        "maxTreeDepths": np.arange(16)+1,
        "skipHeader": False,
    },
}