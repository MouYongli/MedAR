from tokenizers.tools import EncodingVisualizer
from transformers import XLMTokenizer, AutoModel, AutoTokenizer

tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
viz = EncodingVisualizer(tokenizer._tokenizer) # Change here
viz(text="I am a boy")