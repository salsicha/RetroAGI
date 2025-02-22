
# Tokenization is just turing something into an integer


## Train tokenizer

text = "a b c d"

# Step 1: Encode the raw text
raw_bytes = text.encode('utf-8')

# Step 2: Convert the list of bytes into list of integers
tokens = list(map(int, raw_bytes))

# Step 3: Get the most frequent occuring pair from the whole list of frequencies.
def get_stats(tokens):
  stats = {}
  for tok1, tok2 in zip(tokens, tokens[1:]):
    if (tok1, tok2) in stats:
      stats[(tok1, tok2)] += 1
    else:
      stats[(tok1, tok2)] = 1
  return stats
stats = get_stats(tokens)
top_pair = max(stats, key=stats.get)

# Step 4: Replace this with a new token
# Step 5: Repeat steps 3, 4 as long as you want.
vocab_size_final = 276
vocab_size_original = 256
num_merges = vocab_size_final - vocab_size_original
ids = list(tokens) # copy so we don't destroy the original list
for i in range(num_merges):
  stats = get_stats(tokens)
  top_pair = max(stats, key=stats.get)
  idx = vocab_size_original + i
  print(f"merging {top_pair} into a new token {idx}")
  tokens = merge(tokens, top_pair, idx)


## Decoding:

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
vocab[idx] = vocab[p0] + vocab[p1] # Concatenation of bytes objects

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text
  print(decode([128]))



## Encoding

def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    # The above line returns the most eligible pair to be merged and encoded.
    if pair not in merges:
      break # nothing else can be merged

    idx = merges[pair]
    tokens = merge(tokens, pair, idx)

  return tokens