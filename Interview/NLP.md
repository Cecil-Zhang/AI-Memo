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
- Sub-word representations
    - Character-level models: word embeddings are composed from character embeddings
    - Sub-word models: word embeddings are composed from word piece embeddings
        - same architectures as for word-level models, but use smaller units "word pieces"
            - `BPE`
            - `WordPiece`
        - Hybrid architectures: main model has words, something else for characters
    - Advantages: solve OOV problems, better at syntactic
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


### Briefly introduce BPE (Byte Pair Encoding)
- Originally a compression algorithm: most frequent byte pair -> new byte (replace bytes with character ngrams)
- 1. Start with a unigram vocabulary of all (Unicode) characters in data (准备语料，生成字节级词表，确定目标词表大小)
- 2. Most frequent ngram pairs -> a new ngram (统计每一个连续字节对的出现频率，选择最高频者合并为新的subword并加入词表)
- 3. Have a target vocabulary size and stop when you reach it (重复第二步直到达到目标词表大小或下一个最高频率为1)


## Language Models
### What is Language Models
- A system that performs the task of predicting what word comes next. 
- More formally, given sequence of words x_1, x_2, ..., x_t, compute the probability distribution of the next word x_(t+1): P(x_(t+1)|x_t,...,x_1)
- Conditional Language Modeling: the task of predicting the next word, given the words so far y_1,...,y_t, and also some other input x
    - Machine Translation (x=source sentence, y=target sequence)
    - Summarization (x=input text, y=summarized text)
    - Dialogue (x=dialogue history, y=next utterance)


### N-gram Language Models
- Definition: to compute the probabilities mentioned above, the count of each n-gram could be compared against the frequency of each word
- A n-gram is a chunk of n consecutive words. Unigram, bigram, trigram, 4-grams.
- Assumption: x_(t+1) only depends on the preceding n-1 words (Markov Assumption), P(x_(t+1)|x_t,...,x_1)=P(x_(t+1)|x_t,...,x_(t-n+2))=P(x_(t+1),x_t,...,x_(t-n+2))/P(x_t,...,x_(t-n+2))=count(students opened their w)/count(students opened their)


### What's the difference between HMM and N-gram model
- HMM assumes x_t only depends on the previous x_(t-1) and x_t is invisible.
- N-gram assumes x_(t+1) only depends on the preceding n-1 words and x_t is visible.


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
- On each step *t*, there is a hidden state *h_t* and cell state *c_t*
    - Both are vector length *n*
    - The cell state stores long-term information
    - The LSTM can erase, write and read information from the cell
- The selection of which information is erased/written/read is controlled by three corresponding gates
    - The gates are also vector length *n*
    - On each timestep, each element of the gates can be open(1), closed(0),  or somewhere in-between.
    - The gates are dynamic: their value is computed based on the current context
- [Detailed Architectures](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Key to vanishing gradients: LSTM can preserve information over many timesteps via setting forget gate to 1 and input gate to 0
<img src="https://github.com/Cecil-Zhang/AI-Memo/blob/main/img/LSTM-equations.jpg?raw=true" width="500"/>


### GRU (Gated Recurrent Units)
- Motivation: proposed in 2014 as a simpler alternative to the LSTM
- LSTM vs. GRU
    - On each step *t*, LSTM has a hidden state *h_t* and cell state *c_t*, GRU only has a hidden state *h_t*
    - Rule of thumb: LSTM is a good default choice (especially if your data has particularly long dependencies, or you have lots of training data); Switch to GRUs for speed and fewer parameters


### Bidirectional RNNs
- On each step *t*, compute two hidden states for forward and backward RNNs respectively (in general, two RNNs have separate weights)
- Bidirectional RNNs are only applicable if you have access to the entire input sequence.
- If you do have entire input sequence (e.g. any kind of encoding), bidirectionality is powerful (you should use it by default).


### Multi-layer RNNs (Stacked RNNs)
- Stack multiple RNNs
    - The hidden states from RNN layer i are the inputs to RNN layer i+1
- This allows the network to compute more complex representations. 
    - The lower RNNs should compute lower-level features and the higher RNNs should compute higher-level features. 
- High-performing RNNs are often multi-layer (but aren’t as deep as convolutional or feed-forward networks)


### Vanishing/Exploding gradients in RNN
- Gradients get smaller/larger and smaller/larger as it backpropagates further.
- Proof: gradients of RNN invlvoes exponents of the same matrix (RNN shares weight matrix). Thus gradients become smaller/larger, when the matrix is small/large.
- Solution: LSTM for vanishing gradients, gradient clip for exploding gradients.
- *Details at CS224N-Note: Lecture 07*



## State-of-the-art Architectures
### Introduce `Attention`
- Motivation: Information bottleneck problem of seq2seq model - encoding of the source sentence needs to capture all information about the source sentence.
- Core idea: on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence.
- Attention in equation
    1. We have encoder hidden states h_1, ..., h_n
    2. On timestep t, we have decoder hidden state s_t
    3. We get attention socres for this step: e^t=[s_t*h_1, ..., s_t*h_n]  (dot product)
    4. We take softmax to get the attention distribution: \alpha^t=softmax(e^t)
    5. Use \alpha^t to get a weighted sum of the encoder hidden states to get the attention output a^t
    6. Finally we concatenate a^t with s^t and proceed as before
- Advantages
    - Significantly improves NMT performance
    - Solves bottleneck problem
    - Helps with vanishing gradient problem (provides shortcut to faraway states)
    - Provides some interpretability
- General form of attention: given a set of vector **values**, and a vector **query**, **attention** is a technique to compute a weighted sum of the **values**, dependent on the **query**
- Attention Variants
    - Basic dot-product attention
    - Multiplicative attention
    - Additive attention


### Introduce `Transformer`
- Motivation: Learn context representations of variable length data
    - RNN: inhibits parallelization, no explicit modeling of long and short range dependencies, no hierarchy
    - CNN: long-distance dependencies require many layers
    - Why not use attention for representations
- Input Embedding
    - word embeddings + positional embeddings (provides meaningflu distances)
    - List of vectors, size of list is a hyperparameter. Usually the length of the longest sentence.
- Multi-headed Self-Attention
    - Self-Attention: Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.
    - Multi-headed
        - It expands the model’s ability to focus on different positions.
        - It gives the attention layer multiple "representation subspaces".
- Residual
    - Make deep network training healthy
    - Residuals carry positional information to higher layers, among other information
- Decoder side
    - One more sub-layer in the middle compared to encoder side: Encoder-Decoder Attention sub-layer which create its Query matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.
    - The self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to *-inf*) before the softmax step in the self-attention calculation.
- Details at https://jalammar.github.io/illustrated-transformer/


### Introduce `BERT`
- Motivation: Language models only use **left** context or **right** context, but language understanding is bidirectional.
    - Why are LMs are unidirectional?
        - Directionality is needed to generate a well-formed probability distribution
        - Words can "see themselves" in a bidirectional encoder
- BERT contribution: further generalizing unsupervised pre-training to deep **directional** architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.
- Pre-training Task #1: Masked LM (Exploit bidirectional context)
    - Mask out k% (k=15) of the input words, and then predict the masked words. Large k: expensive to train, small k: not enough context.
    - Problem: Mask token never seen at fine-tuning. Solution: don't replace mask token with [MASK] all the time, instead 80% with [MASK], 10% with random word, 10% with original word.
- Pre-training Task #2: Next Sentence Prediction
    - Motivation: many downstream tasks such as QA and NLI are based on understanding the relationship between two sentences, which is not directly captured by language modeling.
    - Predict whether sentence B is actual sentence that proceeds Sentence A.
    - The first token of every sequence is always a special classification token [CLS]. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
- Input representation: each token is sum of three embeddings: token embeddings + segment embeddings + position embeddings.



<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
![alt text](https://github.com/Cecil-Zhang/AI-Memo/blob/main/img/.jpg?raw=true)

