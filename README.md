# StockPricePredictionNLP

# Stock market movement prediction Part 2: Use NLP to predict stock price movement for Tesla with scraped tweets and news 
Implemented sentiment analysis and neural networks (Doc2Vec) for prediction in Python 

Stock market movement prediction is an important issue in the financial world for investment strategies. This project is the part 2 for the "Stock Market Movement Prediction" project which is particularly to analyze the stock price movement for Tesla Inc, but can be expand to other companies by changing the keywords. We want to help them gauge the market sentiment in order to make better investment decisions.

### Prerequisites

This project  is entirely written in Python 3 and needs it to be installed on the computer. 
As this code is generalizable on NLP and sentiment analysis, but a big part of is data preparation which can be fairly specific to this project and needs coding skill to adjust.  
This file is also a python notebook written using Jupyter Notebook. Recommend installing "Anaconda".

### Main parts 
Word2vec is a well-known NLP algorithm, which is often used for short text like tweets. We used a different NLP approach for the news - Doc2vec to process the different format of news’ paragraphs. 
In Doc2vec each paragraph has a unique label. We generated it by bracketing the stock price movement for the next day of the news published date.(see Dependent variables section) Multinomial logistic regression is the algorithm used on training. Both DM model  and CBOW models are tried, the latter has a higher accuracy of 60%. 
The regression equation is: StockPriceMovement_%ChangeBracket=Tagged_Parsed_word
#### Sentiment analysis 
Besides the Doc2vec, we also put the average sentiment of news into a logistic regression model along with data from Twitter to predict the stock price movement for the next day. The accuracy is around 54%. The regression equation is : StockPriceMovement_%ChangeBracket = News_Sentiment_avg+Twitter_sentiment_avg+retweet_count+num_tweets
#### Bonus - WordCloud
We also did qualitative analysis for cleaned news articles and represented them in a form of a word cloud. Though we searched only about Tesla we see the words like shares, stock, investor have high frequency in the news

### Limitation 
There are several limitations of our approach.
1. merging the news/tweets with stock prices makes us lose some tweets/news as stock prices are not there for weekends/holidays. To expand this approach, we recommend trying to find a reasonable way of imputing the stock price on weekends or attributing the news/tweets on weekends to the next Monday as they most likely influence the stock price on the next Monday. 
2. As most of the price movement is around -0.15 ~ 0 which is one of the levels in our dependent variable, it is not a very balanced dataset.
3. There can be other confounding variables/influencers like coronavirus which can influence the whole stock market significantly. It limits our models in predicting the movement correctly. We suggested future models take these confounding variables into account.
4. We only collected 160 days data from twitter and news. As neural network models need large amounts of training data to achieve accuracy, we recommend expanding the data time range or data source to gain more training data.Lastly, because there can be slang/hinglish, we can clean the data more exhaustively.

