# Fake-News

## 1.0	INTRODUCTION
In today’s world, there is a huge increase in the fake news. Studies have shown that this has led to the decrease in the amount of news being consumed by people all around the globe. Moreover, there have been some detrimental consequences from the spread of fake news. People simple just avoid news these days because fact checking can be gruesome and a tedious task. Hence, the aim of our Project is to build a classifier that can predict whether the given snippet of news is real or fake. It can drastically help reduce the time that people can spent on fact checking news. 

### 1.1	Problem statement 
Increase in the spread of fake news have led to a decrease in the amount of news being consumed by people which is important for making well informed decisions.


### 1.2	Data
We took our dataset from Kaggle. Our data source contains 10240 instances and 3 independent variables namely Text, tag and labels. The labels are further classified into 6 categories, each for which we have assigned numerical values:Barely-True – 0, Fake – 1,Half-True – 2, Mostly-True – 3, Not-Known – 4 and True - 5


## 2.0 METHODOLOGY

### 2.1	Data Analysis
#### 2.1.1	Class Distribution
As we can see in the below image, the total number of samples of all the 6 labels are shown. The top one being the “Half True” with 2000+ sample values and the least one being, “not known” with 750 values.

![Class_distribution](https://user-images.githubusercontent.com/90323558/150654993-909c8946-3fce-41f1-8dd2-2623379bd830.jpg)


#### 2.1.2 Topic Modelling using LDA
LDA stands for Latent Dirichlet Allocation. It is an unsupervised algorithm which identifies key topics within a text document and help put together words under those topics based on the frequent occurrences. Under LDA, each document is assumed to have a mix of text, each topic with a certain probability of occurring in the document.
From our Project we have taken some sample text and have put them under topic, after applying LDA.

![LDA](https://user-images.githubusercontent.com/90323558/150655052-0a3dc1a3-8928-4be5-b8c6-615627d11ddf.jpg)

### 2.2	Text Preprocessing
It is important to note that the computer has no idea what it is looking at.  All it sees is a collection of characters.  That is why we need text preprocessing.  Text Preprocessing is a process of preparing our text data to build machine learning models on top of it.  For machine learning models to accept our data as input, our data needs to be in a specific format. 
The first thing we need to do is tokenize our text data. After tokenization, we did six cleansing activities from Stop-words removal to case conversion in our project.

![Text_prepocessing](https://user-images.githubusercontent.com/90323558/150655111-58f50d97-de78-49a2-8f9a-63703d0738ea.jpg)

#### 2.2.1	Word Tokenization
Tokenization is splitting text content into parts, words, and sentences (https://bit.ly/3FRyDjM) .  The most common way is by words.  Converting text into a set of tokens makes it easy to clean the text data further.  We used the NLTK word tokenize method to convert the text data into token lists. 

![token](https://user-images.githubusercontent.com/90323558/150655317-13866b66-d0d1-459d-ad24-00895c438429.jpg)

#### 2.2.2	Stop-words Removal 
Stop-words are words that do not carry any insights.  For example, words like in, and the, and which are abundant in text data.  They are not required for analyzing data; however, they need resources for storage and processing.  So, eliminating most of them is important for efficiency. We ran the previously created token list through a filter function that checks if the token is in the stop-words list. If so, the stop-word is not returned. We finally had the remaining tokens.
 
![stop](https://user-images.githubusercontent.com/90323558/150655329-fd0a3734-ce65-4e92-9b1e-79fb232c7dcf.jpg)

#### 2.2.3	Spelling Correction
Also, we did spell corrections using the spellchecker package for detecting and correcting spelling mistakes.

#### 2.2.4	Lemmatization
Lemmatization is another crucial step in text preprocessing.  It provides a root word that uses a dictionary to match the words.  Lemmatization is a more expensive algorithm than stemming due to the dictionary.  For our project with lemmatization, we used the WordNet Lemmatizer.

![lemma](https://user-images.githubusercontent.com/90323558/150655336-dc9d888f-29cc-4f4a-beab-65dc0b9249d9.jpg)

#### 2.2.5	Contraction Mapping
We see it is easy to combine model verbs with most words or pronouns into a contraction in English.  So, we used a contraction_mapping text file like dictionary mapping to expand the tokens.  For example, “it's” to “it is”. 

![contraction](https://user-images.githubusercontent.com/90323558/150655343-f52a9148-35e1-4424-8129-e1775e0ff389.jpg)

#### 2.2.6	White Spaces Removal
Also, we removed white spaces using the strip function.

#### 2.2.7	Lower Casing
Next, as Case conversion was necessary to standardize text, we converted each token to lowercase using a standard Python function for this purpose. 

![lower](https://user-images.githubusercontent.com/90323558/150655354-f924a5d4-1692-4550-a5d0-627edbafeb2f.jpg)

The results are shown in the picture below.

![clean_text](https://user-images.githubusercontent.com/90323558/150655374-ac12f276-5bbd-42cc-bebe-135abcda3b2a.jpg)


### 2.3	Hyperparameter tuning and model selection:

After we cleaned our data, now we trained our model with different Classifier models (e.g., Random Forest Classifier, Multinomial Naive Bayes, Support Vector Machine, Logistic Regression, Decision Trees. 
In order to find the best model for our project with the best parameters we decided to use Random Search because it would take less time and less computer resources.  For this project we didn’t use Grid Search because could be thought of as an exhaustive search for selecting a model. In Grid Search, the data scientist sets up a grid of hyperparameter values and for each combination, trains a model and scores on the testing data. In this approach, every combination of hyperparameter values is tried which can be very inefficient.(https://bit.ly/3KvCoip)
In addition, we decided to use Count Vectorizer and TF-IDF as a feature extractor.  The difference between those techniques is that TF-IDF not only gives us the frequency of the words but also the importance.
After we trained our model with 2 different techniques of feature extractor. In the next Table , we will see which model was the best. 

|  Model |  Feature_Extractor |  Best_Hyperparameters | AUC Score  |
|---|---|---|---|
| RandomForest |  Tf-IDF |  'model__criterion': ['gini', 'entropy'] 'model__max_depth': [5, 10, 20, 50, 100] 'model__min_samples_leaf': [1, 2, 4]'model__min_samples_split': [2, 5, 10] 'model__n_estimators': [200, 400, 1000]'tfidf__norm': ('l1', 'l2')'tfidf__use_idf': (True, False)| 0.540567809 | 


### 2.4 Simple Neural Network:
#### 2.4.1	Architecture
For our neural network, we used a relatively shallow network of just 2 hidden layers as shown in table below. 

|  Layer|  Neurons|  Activation|
|---|---|---|
| Input|  300 |-|
| Hiden 1| 64|ReLU|
| Hiden 2|128|ReLU|
| Output| 6|Sigmoid|

The input layer took in as input 300 dimensional vectors of embedded values, each representing a single word. The 1st and 2nd Hidden layers have 64 and 128 neurons respectively and along with this they also have the same activation function. The output layer has 6 neurons representing each of our 6 classes and Sigmoid as its activation function. I experimented with Softmax too as the activation for the final layer. However, upon testing I achieved better results through Sigmoid. Furthermore, the network had 2,847,650 parameters in total out of which just 42, 950 were trainable.

#### 2.4.2 GloVe Embeddings

As we have seen above, CountVectorizer and Tf-IDF encodes the words using a single numeric value. However, there is a much more powerful way called GloVe embeddings (https://stanford.io/3FXx7wd) which embeds each word in a 300-dimensional vector.  GloVe has been pretrained on 6 billion words. These vectors when plotted on a multidimensional plane plots closely related words in close proximity to each other.

#### 2.4.3	Text preprocessing for Neural Network
Since the input size to the neural network must be constant, we set 20 as the max length for each sentence. The end of the sentences would be padded with 0’s to match this length. Furthermore, we set a vocabulary size of 6000. Words which are out of this vocabulary range would be replaced with the “<OOV>” (Out of Vocab) token. Finally, the class labels were one-hot encoded.

#### 2.4.4	Training
 
For the loss function we went with binary cross entropy since we are using Sigmoid in the final layer, and we used Adam as our optimizer. The network was trained on 50 epochs with a batch size of 64. Along with this, we also employed early stopping which stops training if it does not see an improvement in the validation loss for 20 epochs.
 
#### 2.4.5	Training Evaluation
 
 <img width="229" alt="training_validation_loss" src="https://user-images.githubusercontent.com/90323558/150656046-9b7acb34-593a-4e01-bd79-06f829958300.png">

The above loss plot denotes that the training loss is decreasing but at the same time the validation loss seems to be increasing through each epoch. One of the major reasons behind this could be attributed to the fact that network is severely overfitting on the train data. We tried to tackle this be implementing early stopping. However, a better approach would be to modify the normalization parameters such as increasing the Dropout rate.
 
### 3.0	RESULTS
 
|  Model|  Feature Extractor|  AUC Score|
|---|---|---|
| Neural Network|  Glove |0.6110|
| Random Forest| Tf-IDF |0.5405|
 
 
Upon evaluating on the test set, as shown in the table abovee, we can see that the simple neural network easily outperformed RandomForest which was our previous best model. We didn’t perform any hyperparameter tuning on the neural network to achieve this.

To test this network further, we tested it on a real fake news headline. The network was able to predict that the given news snippet was fake as shown in the figure below.

 ![detect](https://user-images.githubusercontent.com/90323558/150656202-11177c8b-5336-4b87-b6d0-d0c34b10a8e8.jpg)

 
### 4.0	CONCLUSIONS
To conclude, Tf-IDF performed better than CountVectorizer among the traditional feature extraction methods. Among all the traditional ML algorithms we used, RandomForest with Tf-IDF overall performed the best. The simple 2-layer Neural Network had the best AUC score overall. 
For future work, we think we can further improve the score by training on a bigger dataset. For this project we only had 10K instances to predict 6 labels. Furthermore, we could train on a more advanced network like Transformers using BERT word embeddings. 


