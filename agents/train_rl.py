from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer

max_seq_length = 1024 # Can increase for longer reasoning
lora_rank = 32 # Higher = smarter but more VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True, # Critical for RL speed
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
training_args = GRPOConfig(
    learning_rate = 5e-6,
    lr_scheduler_type = "cosine",
    weight_decay = 0.1,
    max_prompt_length = 256,
    max_completion_length = 768,
    num_generations = 8, # Number of variations to try per prompt
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    max_steps = 250, # RL takes longer than SFT; aim for 300+ steps
    logging_steps = 1,
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [correctness_reward_func, format_reward_func],
    args = training_args,
    train_dataset = dataset, # Your dataset with 'prompt' and 'answer' columns
)

trainer.train()
training_args = GRPOConfig(
    learning_rate = 5e-6,
    lr_scheduler_type = "cosine",
    weight_decay = 0.1,
    max_prompt_length = 256,
    max_completion_length = 768,
    num_generations = 8, # Number of variations to try per prompt
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    max_steps = 250, # RL takes longer than SFT; aim for 300+ steps
    logging_steps = 1,
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [correctness_reward_func, format_reward_func],
    args = training_args,
    train_dataset = dataset, # Your dataset with 'prompt' and 'answer' columns
)

trainer.train()