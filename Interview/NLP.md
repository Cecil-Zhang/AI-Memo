## Word Embeddings
### Word Representing Ways
- As discrete symbols
    - `one-hot encoding`
    - problems: words are  infinite, no natural notion of similarity (one-hot vectors are orthogonal)
- As distributional semantics (by their contexts)
    - A word’s meaning is given by the words that frequently appear close-by (One of the most successful ideas of modern statistical NLP! )
    - When a word w appears in a text, its context is the set of words that appear nearby (within a fixed-size window). 
    - How: count-based vs. shallow window-based
    - Count-based:
        - Rely on matrix factorization, e.g. `LSA`, `HAL`. 
        - Effectively leverage global statistical information and primarily used to capture word similarities
        - Do poorly on tasks such as word analogy
    - Shallow window-based
        - Learn word embeddings by making predictions in local context windows, e.g. `word2vec`
        - Able to capture complex linguistic patterns beyond word similarity
        - Fail to make use of the global co-occurrence statistics
    - `GloVe` (combined both)
- Contextual Word Representations
    - Train RNN on large corpus to predict next words, then stuck `LSTM` layers. These layers can be used to produce context-specific representations.


### Briefly introduce `word2vec`.
- `word2vec (Mikolov et al. 2013)` is a family of algorithms which build dense word vectors by predicting *"context word"* given *"center word"* (or vice versa).
- Idea: 
    - We have a large corpus of text, every word in a fixed vocabulary is represented by a vector
    - Go through each position *t* in the text, which has a *center word c* and *context (“outside”) words o*
    - Use the similarity of the word vectors for c and o to calculate the probability of o given c (or vice versa)
    - Keep adjusting the word vectors to maximize this probability
- Intuition: If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our network is motivated to learn similar word vectors for these two words so that they have similar probabilities.
- Architecture: 
    - Single hidden layer (no `activation function`), but the output layer use `softmax`
    - Why two vectors (the hidden layer and the output layer)? Easier optimization. Average both at the end.
    - Two model variants
        1. Skip-grams (SG): Predict context (”outside”) words (position independent) given center word
            - Works well with small amounts of training data and represents even words that are considered rare
        2. Continuous Bag of Words (CBOW): Predict center word from (bag of) context words.
            - Trains several times faster and has slightly better accuracy for frequent words
- *Details at CS224N-Note: Lecture 01*

### Explain tricks used in `word2vec`, and why do we need them?
- Why: It’s a huge neural network! A lot of training data and a lot of weights to update. For a network with 300 neurons in hidden layer and a vocabulary of 10000 words, we have a weight matrix with 300 x 10000 = 3 million
- Hierarchical softmax to reduce computation cost and subsampling frequent words to decrease the number of training examples.
- Modifying the `optimization objective` with a technique they called “Negative Sampling”, which causes each training sample to update only a small percentage of the model’s weights.
- Hierarchical softmax
    - Use `Binary Huffman Tree` in the output layer instead of flatten layer, as it assigns short codes to the frequent words which results in fast training
    - Reduce computation cost of Probability from N(vocab\_size) to log(vocab\_size)
    - How to build a `Binary Huffman Tree`? Recursively take the least weighted nodes from list as leaves, then put root node back to the list.
- Negative sampling
    - Due to large vocabulary size, every training example will updates tremendous parameter. With negative sampling, we are instead going to randomly select just a small number of “negative” words to update the weights for. We will also still update the weights for our “positive” word.
    - In the hidden layer, only the weights for the input word are updated (this is true whether you’re using Negative Sampling or not).
    - Interesting implementation of negative sampling: They have a large array with 100M elements (which they refer to as the unigram table). They fill this table with the index of each word in the vocabulary multiple times, and the number of times a word’s index appears in the table is given by . Then, to actually select a negative sample, you just generate a random integer between 0 and 100M, and use the word at that index in the table. Since the higher probability words occur more times in the table, you’re more likely to pick those.
- Subsampling frequent words
    - Frequent words like “the”, “a” doesn’t tell much about the context word, and they’re far more than needed
    - For each word we encounter in our training text, there is a chance that we will effectively delete it from the text. The probability (1-sampling rate) that we cut the word is related to the word’s frequency.


### Explain `GloVe`
- `GloVe` stands for Global Vectors for Word Representation. Combining the best of both worlds (count based vs. direct prediction), GloVe consists of a weighted least squares model that trains on global word-word co-occurrence counts and thus make efficient use of statistics.
- Crucial insight: Ratios of co-occurrence probabilities can encode meaning components
- The training objective of GloVe is to learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence. Owing to the fact that the logarithm of a ratio equals the difference of logarithms, this objective associates (the logarithm of) ratios of co-occurrence probabilities with vector differences in the word vector space. Because these ratios can encode some form of meaning, this information gets encoded as vector differences as well. For this reason, the resulting word vectors perform very well on word analogy tasks


### Explain `fastText`. What's the difference compared to `word2vec`?
- A library that allows users to learn text representations and text classifiers.
- Key insight: use the internal structure of a word (sub-word) to improve vector representations obtained from `word2vec`. (sum of vectors of character n-grams and the word itself)
- Comparison with `word2vec`:
    1. Better performance on syntactic word analogy tasks, but degraded performance on semantic analogy tasks.
    2. Slower to train due to added overhead of n-grams.
    3. Better representing out-of-vocabulary words.
    4. Able to classify text. (Simple 1 hidden layer NN, input: sum of input word vectors, output: (hierechical) softmax) 



## Recurrent Neural Networks
### Summary
- Motivation: a neural architecture to deal with any length input
- Variants: vanilla RNN, `GRU`, `LSTM`
- Advantages
    - Can process any length input
    - Computation for step t can (in theory) use information from many steps back
    - Model size doesn’t increase for longer input 
    - Same weights applied on every timestep, so there is symmetry in how inputs are processed.
- Disadvantages
    - Recurrent computation is slow (because it is sequential, it cannot be parallelized)
    - In practice, it’s difficult to access information from many steps back


### LSTM (Long Short-Term Memory)
- Motivation: proposed in 1997 as a solution to vanishing gradients in vanilla RNN
- On each step _t_, there is a hidden state *h_t* and cell state *c_t*
    - Both are vector length *n*
    - The cell state stores long-term information
    - The LSTM can erase, write and read information from the cell
- The selection of which information is erased/written/read is controlled by three corresponding gates
    - The gates are also vector length *n*
    - On each timestep, each element of the gates can be open(1), closed(0),  or somewhere in-between.
    - The gates are dynamic: their value is computed based on the current context
- Architectures
![LSTM equations](https://github.com/Cecil-Zhang/AI-Memo/img/LSTM-equations.jpg?raw=true)
![LSTM gates](https://github.com/Cecil-Zhang/AI-Memo/img/LSTm.jpg?raw=true)

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
![alt text](https://github.com/Cecil-Zhang/AI-Memo/img/.jpg?raw=true)

