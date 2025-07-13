
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    
def generate_text(model_path: str,prompt, max_length=100):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(torch.device("cuda"))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser() #description="Run text generation using a Causal Language Model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory.")
    args = parser.parse_args()

    # Example text generation
    prompt = "What are the applications of quantum computing?"
    print("\nGenerated Text:")
    print(generate_text(args.model_path,prompt))

    from datasets import load_dataset

    dataset = load_dataset(
        path=args.data_path, 
        name="wikitext-2-raw-v1",
        split="train[:10%]"
    )
    # Add padding token to the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Define the tokenizer function with a fixed max_length
    def tokenize_function(examples):
        encoding = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        encoding["labels"] = encoding["input_ids"].copy()  # Assign labels, required to avoid training errors for this example
        return encoding


    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define Trainer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(torch.device("cuda"))

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Start fine-tuning (this will take time)
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
