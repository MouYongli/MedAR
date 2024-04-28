from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from transformers import EncoderDecoderModel
from tokenizers import Tokenizer, models
from tokenizers.processors import TemplateProcessing
import torch
import numpy as np
import nltk
from torch.utils.data import DataLoader
import pandas
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import argparse

loaded_tokenizer = AutoTokenizer.from_pretrained('medal_small_tokenizer')

def tokenize_function(example):
    tokenized_batch = loaded_tokenizer(example["MASKED_TEXT"], example["ABBR"], truncation='only_first', padding='max_length', max_length=512)
    #print(example["abbr"])
    #print(example["masked_text"][0])
    #print('input: ', tokenized_batch["input_ids"][0])
    with loaded_tokenizer.as_target_tokenizer():
        #print(example["LABEL"])
        label = example["LABEL"]
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
    length = len(example['LOCATION'])
    for i in range(length):
        mask_list = []
        ids = example['LOCATION'][i]
        #print("ids: ", ids[0])
        text = example['TEXT'][i].split()
        #print("text: ", text)
        abbr = text[int(ids)]
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
        if i % 10000 == 0:
            print(i)
    example['ABBR'] = ab
    example['MASKED_TEXT'] = masked_text
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


def main(args):
    device = args.device
    checkpoint = args.checkpoint
    #model = RobertaForCausalLM.from_pretrained(checkpoint, config=config)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(checkpoint, checkpoint)
    model.to(device)

    
    loaded_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.decoder_start_token_id = 1
    model.config.eos_token_id = 2
    model.config.pad_token_id = 3


    loaded_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    train_data = pandas.read_csv('../../../../llm-research/MeDAL/pretrain_subset/train.csv')[:1000]
    test_data = pandas.read_csv('../../../../llm-research/MeDAL/pretrain_subset/test.csv')[:1000]
    #small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    #small_val_dataset = dataset["validation"].shuffle(seed=42).select(range(1000))
    #small_test_dataset = dataset["test"].shuffle(seed=42).select(range(100))

    data_train = mask_make(train_data)
    data_test = mask_make(test_data)
    #print(small_train.column_names) #['abstract_id', 'text', 'location', 'label', 'abbr']
    #encoding = loaded_tokenizer.encode(small_train[0]['text'], small_train[0]['abbr'])
    #print(encoding.tokens)
    train_dataset = Dataset.from_pandas(data_train).shuffle(args.seed)
    test_dataset = Dataset.from_pandas(data_test).shuffle(args.seed)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
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
        output_dir= args.output_dir,
        save_steps = 10000,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='FacebookAI/roberta-base')
    parser.add_argument('--output_dir', default='output/train')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
