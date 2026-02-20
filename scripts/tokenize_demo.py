from llm_scratch.data.tokenizer import SimpleTokenizerV1
import re

with open("data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read();

preprocessed = re.split(r'([,.?!_"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))

vocab = {token:integer for integer,token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab)

text = """It's the last the painted, you know," Mrs. Gisburn said"""
ids = tokenizer.encode(text)

print(ids)
print(tokenizer.decode(ids))

