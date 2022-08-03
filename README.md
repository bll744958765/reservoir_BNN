# A reliable Bayesian Neural Network for the prediction of reservoir thickness with quantified uncertainty
Li-Li Bao, Jiang-She Zhang, Chun-Xia Zhang, Rui Guo, Xiao-Li Wei, Zi-Lu Jiang

**This code implemented by Li-Li Bao for the prediction of reservoir thickness with quantified uncertainty is based on PyTorch 1.11.0, python 3.8.
GPU is NVIDIA GeForce RTX 3080.

The structure of our BNN:

![structure1](https://user-images.githubusercontent.com/92556725/182527690-4ec3edb5-e06c-4cdd-8c0a-06fb31229288.jpg)

## Getting Start for the prediction of reservoir thickness with quantified uncertainty

##Run the code for the predict of reservoir thickness 


If you would like to try predict examples, you can download the train.cvs, valid.cvs and prediction.cvs and run prediction.py in Windows 64bit. Where the train.cvs is used to train our BNN, and the valid.cvs is used to test the presiction effection. The prediction.cvs is a dataset need to been predict.

In prediction.py, you only need to set the loading paths path1, path2 and path3 as your paths of these three files to run the program.
## Dataset
In fact, the original training data set includes 459 logging data, and each logging is given an ID number. We divided the data set into 10 parts, of which 9 is used to train our network, and 1 is used as a test set to verify the effect of the network. Therefore, the training set and the test set have the same structure. They include 14 auxiliary variables such as Line, CMP, Freq, and so on. Please refer to the manuscript for the actual significance of auxiliary variables. SValue represents the reservoir thickness value, which is our prediction variable. SValue in training set and test set is known.

![image](https://user-images.githubusercontent.com/92556725/182522672-8e05800b-6891-4d00-a47a-4ad668762bba.png)


Since our real data is confidential, we modified the data vlues during  uploaded the data set, but this does not affect the operation of the network. This leads to the difference between your running results and the description in the manuscript.

![131139](https://user-images.githubusercontent.com/92556725/182529230-dd6961ee-30b9-4e32-8b47-c4fb628a410c.png)


More details can be found in manuscript.
