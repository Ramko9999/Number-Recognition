# Number-Recognition
I am doing this project to get my feet wet with machine learning. What better way to get introduced to machine learning than implementing different algorithms to classify numbers.

I would like to thank Kensata(he has a github) for providing free training data. I trained a fairly unsophisticated model. It has 900 nodes for the input layer, 30 for the hidden layer and 10 for the output layer. By using FMINCG, I was able to get my model to converge with a strong accuracy for the training phase. However, in terms of test accuracy, it underperformed a bit, as the test accuracy hovered over 70%. I think the test accuracy is expected as the model itself might be too simple for such a problem. 

In the near future, I will experiment with different layerd neural nets as well as different activation functions & so forth to get a sense of what works.

As of 4/19/2019, I changed the number of hidden units from 30 units to 100 just to see what it would do. My testing accuracy went up to 83%. That shit is crazy man!

As of 4/20/2019, I changed the number of hidden from 100 to 300, and it go to 86% accuracy on a different randomized testing dataset. I fed the model 10 of my own digits, it was able to get 3 of them correct. I think I might need to right the digits bit boldy due to the training data.
