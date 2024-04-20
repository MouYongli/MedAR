from tokenizers.tools import EncodingVisualizer
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

#tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-13b")
dataset = load_dataset("medal")

context = "to facilitate the proper and quick DUE of cancer chemotherapy the issues of surrogate and true endpoints in clinion of renal CF in patients afflicted with iga nephropathy but normal renal CF urinary excretion of da correlated positively with bp responses to changes from to mmolday salt intake in black SR sr normotensives nt and sr hypertensives under LS NI mmolday but not ssnt and ssht the saline infusion induced increments of da and dopac urinary excretion correlated significantly with increments of sodium urinary excretion and sodium fractional excretion patients afflicted with heart failure hf have a reduced delivery of LD to the kidney accompanied by an increase in daldopa urinary ratios this suggests that hf patients have an increased ability to take up or decarboxylate LD sodium restriction resulted in a significant decrease in urinary LD da and dopac in hf patients suggesting that the system responds to sodium it is concluded that activity of renal DA system may be altered in ss subjects despite the level of their bp and an enhanced delivery of LD to the kidney may be beneficial in edema formation states"
#a = vis(text = context, default_to_notebook = False)

#with open('vis_token.html', 'w') as f:
    #f.write(a)
#def tokenize_function(examples):
    #return tokenizer(examples["text"], padding="max_length", truncation=True)


small_train_dataset = dataset["train"].shuffle(seed=42).select(range(50000))
small_val_dataset = dataset["validation"].shuffle(seed=42).select(range(1000))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))
#tokenized_datasets = small_train_dataset.map(tokenize_function, batched=True)



tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = Whitespace()

with open("train.txt", "w") as output:
    output.write(" ".join(small_train_dataset['text']))

with open("val.txt", "w") as output:
    output.write(" ".join(small_val_dataset['text']))

with open("test.txt", "w") as output:
    output.write(" ".join(small_test_dataset['text']))

tokenizer.train(files=["train.txt", "val.txt", "test.txt"], trainer=trainer)

my_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

my_tokenizer.save_pretrained("../medal_small_tokenizer")

loaded_tokenizer = AutoTokenizer.from_pretrained("../medal_small_tokenizer")

vis  = EncodingVisualizer(loaded_tokenizer._tokenizer)
a = vis(text = context, default_to_notebook = False)

with open('vis_medal.html', 'w') as f:
    f.write(a)
