# A reliable Bayesian Neural Network for the prediction of reservoir thickness with quantified uncertainty
Li-Li Bao, Jiang-She Zhang, Chun-Xia Zhang, Rui Guo, Xiao-Li Wei, Zi-Lu Jiang

This code implemented by Li-Li Bao for the prediction of reservoir thickness with quantified uncertainty is based on PyTorch 1.11.0, python 3.8.

GPU is NVIDIA GeForce RTX 3080.

## The structure of our BNN:

![structure1](https://user-images.githubusercontent.com/92556725/182527690-4ec3edb5-e06c-4cdd-8c0a-06fb31229288.jpg)

## Getting Start for the prediction of reservoir thickness with quantified uncertainty

Run the code for the predict of reservoir thickness :


If you would like to try predict examples, you can download the train.csv and prediction.csv and run prediction.py in Windows 64bit. Where the train.csv is used to train  and test our BNN prediction effection. The prediction.csv is a dataset need to been predict.

In order to run the code, you only need to set the loading paths path1  and path2 in prediction.py as your paths of these two files to run the program.

## Dataset
In fact, the original training data set includes 459 logging data, and each logging is given an ID number. We divided the data set into 10 parts, of which 9 is used to train our network, and 1 is used as a test set to verify the effect of the network. Therefore, the training set and the test set have the same structure. They include 14 auxiliary variables such as Line, CMP, Freq, and so on. Please refer to the manuscript for the actual significance of auxiliary variables. SValue represents the reservoir thickness value, which is our prediction variable. SValue in training set and test set is known.

![image](https://user-images.githubusercontent.com/92556725/182534564-8e0b9437-cafc-42e4-ba5d-b2622adc8285.png)


## Prediction 
The map of reservoir thickness and uncertainty on the predicrion data set:
![prediction  and uncertainty map](https://user-images.githubusercontent.com/92556725/182537339-492f6845-5f75-46a5-8cb7-9745b3862c15.jpg)


## Other
Since our real data is confidential, we modified the data set before uploaded them, this does not affect the run of the code. This leads to the difference between your running results and the description in the manuscript.

More details can be found in manuscript.
