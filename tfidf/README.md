#TF-IDF Results
Running the following will generate a file called tfidf_weights.txt, which is intended to be consumed by the rapid miner process in tfidf_rapidminer.rmp. 
You will need to update the path in the first step of the process
\>  python tfidf_feature_generator.py

In the rapid miner process I undersampled the nonDRNA class to 500 records for training, leaving the other classes. I also used 10-fold cross validation unless noted. I evaluated four tests:

1) kNN classifier on the features in tfidf_weights.txt for original labels
     accuracy: 61.84% +/- 4.79% (micro average: 61.84%)
2) kNN classifier on the features in tfidf_weights.txt for binary classification of IsDNA
     accuracy: 75.42% +/- 2.40% (micro average: 75.42%)
3) kNN classifier on the features in tfidf_weights.txt for binary classification of IsRNA
     accuracy: 76.88% +/- 3.04% (micro average: 76.88%)
4) Trained model from (1) applied to the entire original dataset
     accuracy: 68.24%

##TODO
- Make this into a function which returns feature vectors that can be passed into the neural network