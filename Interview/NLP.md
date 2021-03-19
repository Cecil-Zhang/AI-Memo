## Word Embeddings
### Briefly introduce `word2vec`.
- `word2vec (Mikolov et al. 2013)` is a family of algorithms which build dense word vectors by predicting *"context word"* given *"center word"* (or vice versa).
- Intuition: If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our network is motivated to learn similar word vectors for these two words so that they have similar probabilities.
- Objective function: $J(\theta)=-\frac{1}{T}logL(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\sum_{-m\le j \le m\;(j \ne 0)}logP(w_{t+j}|w_t;\theta)$

