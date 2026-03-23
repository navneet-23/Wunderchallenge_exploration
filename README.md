# Wunderchallenge_exploration
My analysis of wunder challenge 2026

The notebooks I have added are my attempts at solving the wunder challenge. The processes might seem random in the notebook as they are all experimental stuff I was doing to get some idea about the data. I have tried summarizing everthing in this readme file and added the notebooks in order to see the outputs of my work

#**INTRODUCTION TO THE COMPETITION**
I was given anonymized orderbook data which comprised of 6 columns for bid prices , 6 columns for ask prices, 6 columns for bid volumes, 6 columns for ask volumes, 4 columns for traded price and 4 columns for traded volumes. All this data was anonymized and hence made the challenge very difficult. Other than these columns, we were provided with seq_ix and step_in_seq which was a result of the breaking down time series order data into multiple sequences and shuffling them. Each sequence comprised of 1000 steps where the last 900 steps were to be predicted. The goal was to predict two indicators t1 and t0, again anonymized. I was givwen a train dataset of over 10 million rows. The objective metric was weighted_pearson_correlation where the weights were simply the magntitude of the target variable.

**Approaches**
Before trying to use any model on the dataset directly, I performed extensive EDA on all features and targets. While most features and targets showed very low correlations (close to 0.2), a particular correlation between p5 - p11 and the delta of target varible showed a correlation of 0.8. This discovery led me to modelling of delta of target variable instead of target variable directly. While going through this method I faced a lot of challenges that I will describe in the challenges section, but the end result was that direct delta target modelling was not feasiable.
My next approach was to use a deep learning model like GRU. The choice was natural as the data was a time series one and LSTM would be an overkill for this problem. GRU gave me a baseline to start from, without any feature extraction I got a weighted_pearson_correlation of 0.35 in t0 ( target 0) and 0.13 in t1 (target 1). Clearly, a much better apporach was required in case of t1. 
To further enhance GRU, I realized I do not have a lot of hyperarameters to play with, so as a parallel model, I tried modelling a TCN ( Temporal Convolutional Neural Network). Now I had control over layers, activations functions and batch sizes. By some trial and error I identified a good batch size and number of layers. But the major gains were not from these small tweaks
I found a much higher gain when I decided to modify the loss function. So when I was training a vanilla gru or a tcn, my loss function by default was MSE which was being minimized, but I realized that weighted correlation is different than MSE. So I wrote a different loss function that was a hybrid combination of inverse of correlartion and MSE. This led my t1 correlation to jump to 0.16

**Challenges in Linear Regression and Delta t1**
When i modelled a simple linear regression with delta t1, I got a very high R2 score, of close to 0.98 in validation data. This showed that a clear signal existed, but the moment I tried reconstructing t1 using cumsum function, the outputs deviated. This was because there was a constant bias ( either positive or negative) in all sequences which was very small but accumulated in 1000 steps. Thus it gave a very bad global correlation even though individually correlations were high. Infact the average of groupwise correlation was 0.31 for t1 (I really wish I could have got it globally). 
To remove this constant bias, I tried removing the mean of my predictions from the sequences, but **there was a fatal flaw**. *I did not realize that while I am removing the mean, I am visiting the future and correcting myself in past, which is simply not allowed!*
Other than this issue, the problem in predicting delta of t1 was that I never knew the starting point. Intitally I assumed that starting point or the anchor would not be an issue simply because correlation is invariant of shifts or multiples in data. But I soon realized that when you see global data, there are multiple anchors at the start of every sequence, which if predicted wrong give me worse results.
Due to these two issues I dropped the idea of modelling t1 using delta t1 even though a signal existed.

**Final Solution**
My final solution comprised of TCN modelling in t1 and GRU modelling for t0 with hyperparameters determined from grid search. It definetly did not feel whole because I was missing out on using the signal I found, but yeah, I had no way of removing that bias without looking ahead into the future. Finally I got a weighted_correlation average of 0.29


