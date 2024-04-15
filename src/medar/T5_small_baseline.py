from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, BertForMaskedLM, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import torch
import numpy as np
import nltk
import datasets
from torch.utils.data import DataLoader
nltk.download('punkt', quiet=True)
metric = datasets.load_metric("rouge")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)
checkpoint = "google-t5/t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.to(device)
dataset = load_dataset("medal")
loaded_tokenizer = AutoTokenizer.from_pretrained("medal_small_tokenizer")
loaded_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(example):
    tokenized_batch = loaded_tokenizer(example["masked_text"], example["abbr"], truncation='only_first', padding='max_length', max_length=512)
    #print(example["abbr"])
    #print(example["masked_text"][0])
    #print('input: ', tokenized_batch["input_ids"][0])
    with loaded_tokenizer.as_target_tokenizer():
        #print(label[0]) ['muscle length']
        label = [x for xs in example["label"] for x in xs]
        #print(label)
        #res = '' in label
        #print("Is any string empty in list? : " + str(res))
        target_encodings = loaded_tokenizer(label, max_length=10, truncation=True)
        
    #print(target_encodings)
    
    return {"input_ids": tokenized_batch["input_ids"], 
           "attention_mask": tokenized_batch["attention_mask"], 
           "labels": target_encodings["input_ids"]}

def mask_make(example):
    #print(range(len(example)))
    ab = []
    masked_text = []
    #label_ids = []
    for i in range(len(example['location'])):
        mask_list = []
        ids = example['location'][i]
        #print("ids: ", ids[0])
        text = example['text'][i].split()
        #print("text: ", text)
        abbr = text[int(ids[0])]
        #print("abbr: ", abbr)
        #abbr_chr = abbr.split()
        #for j in range(len(abbr_chr)):
            #mask_list.append("[MASK]")
        text = list(map(lambda x: x.replace(abbr, '[MASK]'), text))
        #label_id = loaded_tokenizer(example['label'][i])["input_ids"]
        #print(text)
        #print(" ".join(text))
        ab.append(abbr)
        masked_text.append(" ".join(text))
        #label_ids.append(label_id)
    example = example.add_column("abbr", ab)
    example = example.add_column("masked_text", masked_text)
    #example = example.add_column("label_id", label_ids)
    #print(example)
    return example


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = loaded_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, loaded_tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = loaded_tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

loaded_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
#small_val_dataset = dataset["validation"].shuffle(seed=42).select(range(1000))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(100))

small_train = mask_make(small_train_dataset)
small_test = mask_make(small_test_dataset)
#print(small_train.column_names) #['abstract_id', 'text', 'location', 'label', 'abbr']
#encoding = loaded_tokenizer.encode(small_train[0]['text'], small_train[0]['abbr'])
#print(encoding.tokens)


tokenized_datasets = small_train.map(tokenize_function, batched=True, remove_columns=small_train.column_names)
tokenized_test_datasets = small_test.map(tokenize_function, batched=True, remove_columns=small_test.column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer=loaded_tokenizer, model=checkpoint)
columns = ['input_ids', 'labels', 'input_ids','attention_mask'] 
tokenized_datasets.set_format(type='torch', columns=columns)
tokenized_test_datasets.set_format(type='torch', columns=columns)

#['abstract_id', 'text', 'location', 'label', 'input_ids', 'token_type_ids', 'attention_mask']"
#tokenized_datasets = tokenized_datasets.remove_columns(['abstract_id', 'text', 'location', 'abbr', 'masked_text'])
#print(tokenized_datasets.column_names)
#tokenized_datasets = tokenized_datasets.rename_column("label_id", "label")
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#tokenized_datasets.set_format("torch")
#print(tokenized_datasets.column_names) ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
#tokenized_test_datasets = tokenized_test_datasets.remove_columns(['abstract_id', 'text', 'location', 'abbr', 'masked_text'])
#tokenized_test_datasets = tokenized_test_datasets.rename_column("label_id", "label")
#tokenized_test_datasets = tokenized_test_datasets.rename_column("label", "labels")
#tokenized_test_datasets.set_format("torch")

'''test = small_train[1]
inputs = loaded_tokenizer(test["masked_text"], test["abbr"], return_tensors="pt").to(device)
#print(inputs)
with torch.no_grad():
    logits = model(**inputs).logits
# retrieve index of [MASK]
#print(inputs.input_ids)
#print(loaded_tokenizer.mask_token_id)
#loaded_tokenizer.mask_token_id = 4
#mask_token_index = (inputs.input_ids == 4)[0].nonzero(as_tuple=True)[0]
#predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
print(loaded_tokenizer.decode(predicted_token_id))'''

train_dataloader = DataLoader(
    tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_test_datasets, batch_size=8, collate_fn=data_collator
)

training_args = Seq2SeqTrainingArguments(
    output_dir="bert_filling_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_test_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

