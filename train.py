import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import os

# Load the BookSum dataset
dataset = load_dataset("pszemraj/booksum-short", split='train')
eval_dataset = load_dataset("pszemraj/booksum-short", split='validation')

# Load the pre-trained model and tokenizer
hf_tag = "pszemraj/led-base-book-summary"
model = AutoModelForSeq2SeqLM.from_pretrained(hf_tag)
tokenizer = AutoTokenizer.from_pretrained(hf_tag)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = examples['chapter']  
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary_text'], max_length=128, truncation=True)  
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',       
    evaluation_strategy="epoch",    
    learning_rate=2e-5,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8, 
    num_train_epochs=3,              
    weight_decay=0.01,               
    save_total_limit=1,              
    save_steps=10_000,               
    logging_dir='./logs',            
    logging_steps=200,               
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=tokenized_datasets,    
    eval_dataset=tokenized_eval_datasets,     
    data_collator=data_collator,         
)

# Train the model
trainer.train()

save_dir = os.path.join(os.path.dirname(__file__), 'fine_tuned_model')
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)



