import numpy as np

from tqdm.auto import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import evaluate
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer



from .tfidf_clf import save_evaluation_results

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_n_samples_per_category(df, category_col_name, n_sample):
    # Ensure that the specified category column exists in the DataFrame
    if category_col_name not in df.columns:
        raise ValueError(f"Column '{category_col_name}' not found in the DataFrame.")

    # Shuffle the DataFrame rows
    df_shuffled = df.sample(frac=1)

    # Define a function to get n_sample rows for each category
    def get_samples(group):
        return group.head(n_sample)

    # Use groupby and apply to get n_sample rows for each category
    result_df = df_shuffled.groupby(category_col_name, group_keys=False, sort=False).apply(get_samples)

    del df_shuffled
    return result_df

def get_hf_datasets(df, data_col, target_col, split_ratio=0.8):
    # Create a dictionary with the DataFrame columns
    dataset_dict = {
        "text": df[data_col].tolist(),
        "label": df[target_col].tolist(),
    }

    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_dict(dataset_dict)

    train_dataset = hf_dataset.train_test_split(test_size=1 - split_ratio)["train"]
    val_dataset = hf_dataset.train_test_split(test_size=1 - split_ratio)["test"]
    return train_dataset, val_dataset

def get_tokenized_datasets(tokenizer, train_dataset, val_dataset):
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    hf_train_tkn = train_dataset.map(preprocess_function, batched=True)
    hf_val_tkn = val_dataset.map(preprocess_function, batched=True)
    return hf_train_tkn, hf_val_tkn

def get_train_objects(df, data_col, target_col):
    # Use LabelEncoder to convert string labels to numerical values
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df[target_col])
    train_dataset, val_dataset = get_hf_datasets(df, data_col, "label_encoded")

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    hf_train_tkn, hf_val_tkn = get_tokenized_datasets(tokenizer, train_dataset, val_dataset)

    # Get id2label and label2id mappings from LabelEncoder
    id2label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}

    num_class = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=num_class, id2label=id2label, label2id=label2id
    )
    return model, tokenizer, hf_train_tkn, hf_val_tkn

def train_distilbert(df, data_col, target_col, save_dir, epoch):
    model, tokenizer, hf_train_tkn, hf_val_tkn = get_train_objects(df, data_col, target_col)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epoch,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to=[],  # Set this to an empty list to disable wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_tkn,
        eval_dataset=hf_val_tkn,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def test_distilbert(df_test, data_col, target_col, save_dir, checkpoint_num, txt_save_dir):
    texts = df_test[data_col].tolist()
    true_labels = df_test[target_col].tolist()
    print("Tokenizer creating")
    tokenizer = AutoTokenizer.from_pretrained(f"{save_dir}/checkpoint-{checkpoint_num}")
    print("Tokenizing start")
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    print("Tokenizing end")
    # Ensure that the model is on the same device as the inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # Move input tensors to the specified device
    encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}
    print("input tensor moved to :", device)


    # Load the model on the same device
    classifier = pipeline("sentiment-analysis", model=f"{save_dir}/checkpoint-56980", device=device)

    # Perform batch inference
    batch_size = 128  # Adjust the batch size based on your system's capabilities
    num_samples = len(texts)
    y_test_pred = []
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_inputs = {key: val[i:i+batch_size] for key, val in encoded_inputs.items()}
        
        # Move batch inputs to the specified device
        batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}
        
        batch_results = classifier.model(**batch_inputs)

        # Extract labels and probabilities
        logits = batch_results.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        labels = torch.argmax(probabilities, dim=-1).tolist()
        y_test_pred.extend([classifier.config.id2label[l] for l in labels])


    
    save_evaluation_results(true_labels, y_test_pred, txt_save_dir, "hfdistilbert")