import numpy as np
from math import log

# A mock function to represent the NMT model step, for demonstration purposes.
def nmt_model_step(sequence, vocab_size):
    # Normally this would involve a neural network to predict the next token probabilities
    # Here we just return a fixed probability distribution for simplicity
    return np.random.dirichlet(np.ones(vocab_size), size=1).flatten()

def beam_search_decoder(nmt_model_step, k, vocab_size, max_length=50):
    sequences = [[[], 0.0]]
    # iterate until all sequences end with the <eos> token or reach max length
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq and seq[-1] == '<eos>':
                # If the sequence already ended, add it without expanding
                all_candidates.append((seq, score))
                continue
            next_probs = nmt_model_step(seq, vocab_size)
            for j, prob in enumerate(next_probs):
                candidate = [seq + [j], score - log(prob)]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
        # Check if all sequences ended
        if all(seq[-1] == '<eos>' for seq, _ in sequences):
            break
    return sequences

# Get user input for the vocabulary
vocab_size = int(input("Enter the vocabulary size: "))
index_to_token = {}
for i in range(vocab_size):
    token = input(f"Enter token {i}: ")
    index_to_token[i] = token

# Add special tokens
if '<eos>' not in index_to_token.values():
    index_to_token[vocab_size] = '<eos>'
    vocab_size += 1

# Get user input for the beam width
beam_width = int(input("Enter the beam width: "))

# Decode sequence
result = beam_search_decoder(nmt_model_step, beam_width, vocab_size)

# Print result with tokens
for seq, score in result:
    token_seq = [index_to_token[idx] for idx in seq]
    print(token_seq, score)