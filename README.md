# English-French-Translator

This project demonstrates an end-to-end neural machine translation model using deep learning techniques to translate English sentences into French. The model is built using a Bidirectional LSTM network to learn the relationship between English and French sentences and produce accurate translations.

## Project Overview

This project utilizes a neural machine translation model built with TensorFlow and Keras. The model processes English sentences and outputs their French translation, using techniques such as tokenization, padding, and LSTM layers. The primary goal is to explore the capabilities of sequence-to-sequence models for language translation and to improve translation quality through model tuning.

## Dataset

This project uses a parallel corpus of English and French sentences. The dataset contains 137,859 sentence pairs, providing a substantial amount of data for training the model. The sentences are tokenized and preprocessed to ensure the model learns both the linguistic structure and vocabulary of the two languages.

## Model Architecture

The model follows a sequence-to-sequence architecture, using the following components:

1. **Embedding Layer**: Transforms words into dense vectors that represent their meaning.
2. **Bidirectional LSTM Layer**: Processes the input sequence in both directions, capturing context from both the past and future of a given word.
3. **RepeatVector**: Replicates the encoded input sequence for each timestep of the output sequence.
4. **LSTM Layer**: Generates the output sequence based on the input encoding.
5. **TimeDistributed Layer**: Applies a dense layer to each timestep, outputting the predicted words for each timestep in the sequence.
6. **Softmax Activation**: Ensures the output consists of probabilities, with each word representing the likelihood of a translation.

## Training

The model is trained with the following parameters:

- **Batch Size**: 64
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-Entropy

The training involves feeding English sentences into the model and teaching it to output the corresponding French sentences. Throughout the epochs, the model’s accuracy improves, as shown by decreasing loss values and increasing validation accuracy.

## Results

After 10 epochs, the model achieved an accuracy of **99.07%** on the training data and **98.21%** on the validation data. This shows strong generalization, with the model performing well on unseen data.

Example translation:

- **English**: "I am going to Paris"
- **French**: "Vais-je à Paris"

While the model performs well in most cases, there are areas for improvement in handling certain sentence structures and nuances.

## Improvements

While the current model performs well, several improvements can be made to further enhance its accuracy and robustness:

1. **Increase LSTM Units and Embedding Dimension**: Increasing the number of LSTM units from 256 to 512 and the embedding dimension from 256 to 512 could give the model more capacity to capture complex relationships.
   
2. **Hyperparameter Tuning**: Optimizing the learning rate, batch size, and other hyperparameters could lead to better performance.
   
3. **Pre-trained Embeddings**: Using pre-trained word embeddings like GloVe could enrich the model’s understanding of words and their relationships.
   
4. **Layer Stacking**: Adding more LSTM layers may improve the model’s ability to capture more intricate patterns, though it increases computational cost.
