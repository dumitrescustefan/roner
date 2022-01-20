# RONER

API:
import roner

ner = roner.NER(model="", gpu="", batch=1, window_size=512, named_persons_only=False)

input_texts = ["Ana are mere. Maria are pere.", "b", "c"]

outputs = ner(inputs, show_pbar=False)

for input_text, output in zip(input_texts, output):
  print(input_text)
  for token in output.tokens:
    print(token.text, token.label)
  
