# VN-Accent-Restorer

This project applies multiple deep learning models to the pointless problem of restoring diacritical marks to sentences in Vietnamese.

E.g: ```do an dem trang beo``` → ```đồ ăn đêm trang béo```

## Beamsearch and Language Model

Using a language model, we basically try to guess the next word in the sequence using the previously decoded words to obtain one with the lowest perplexity.
Since using a pure greedy algorithm will make us unable to correct a past mistake, beamsearch will be used along with it.

In this project I implemented two kind neural language models. The traditional RNN model [cite] and a modified version of the Transformer model, originally used 
for sequence to sequence tasks.
![Transformer vs Transformer-Decoder](img/img1.png "Title")
## Sequence to Sequence Model
The same Transformer-Decoder can be used as a end-to-end solution to this problem by modifying the input and training target a bit.

