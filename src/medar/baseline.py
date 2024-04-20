from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from transformers import EncoderDecoderModel
from tokenizers import Tokenizer, models
from tokenizers.processors import TemplateProcessing
import torch
import numpy as np
import nltk
import datasets
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = "FacebookAI/roberta-base"

#model = RobertaForCausalLM.from_pretrained(checkpoint, config=config)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(checkpoint, checkpoint)
model.to(device)

dataset = load_dataset("medal")
loaded_tokenizer = AutoTokenizer.from_pretrained("medal_small_tokenizer")
loaded_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.decoder_start_token_id = 1
model.config.eos_token_id = 2
model.config.pad_token_id = 3

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
    decoded_preds = loaded_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, loaded_tokenizer.pad_token_id)
    decoded_labels = loaded_tokenizer.batch_decode(labels, skip_special_tokens=True)
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_1= sentence_model.encode(decoded_preds, convert_to_tensor=True)
    embedding_2 = sentence_model.encode(decoded_labels, convert_to_tensor=True)
    results = util.pytorch_cos_sim(embedding_1, embedding_2)
    
    return {
        'similarity': torch.mean(results)
    }

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


tokenized_datasets = small_train.map(tokenize_function, batched=True, remove_columns=small_train.column_names)
tokenized_test_datasets = small_test.map(tokenize_function, batched=True, remove_columns=small_test.column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer=loaded_tokenizer, model=checkpoint)
columns = ['input_ids', 'labels', 'input_ids','attention_mask'] 
tokenized_datasets.set_format(type='torch', columns=columns)
tokenized_test_datasets.set_format(type='torch', columns=columns)

train_dataloader = DataLoader(
    tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_test_datasets, batch_size=8, collate_fn=data_collator
)

training_args = Seq2SeqTrainingArguments(
    output_dir="saved_model",
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

