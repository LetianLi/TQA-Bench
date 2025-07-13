import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm

class QwenTransformersChat:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None, max_context=8192):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.max_context = max_context
        self.messages = []  # List of {"role": str, "content": str}

    def reset(self):
        self.messages = []

    def append_turn(self, role, content):
        self.messages.append({"role": role, "content": content})

    def _build_inputs(self, add_generation_prompt=True):
        # Use chat template for prompt formatting
        return self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        ).to(self.device)

    def _count_tokens(self, input_ids):
        return input_ids.shape[-1]

    def _truncate_history(self):
        # Remove oldest turns until context fits
        while True:
            input_ids = self._build_inputs()
            n_tokens = self._count_tokens(input_ids)
            if n_tokens <= self.max_context:
                break
            if len(self.messages) > 1:
                self.messages.pop(0)
            else:
                break

    def chat(self, user_input, max_new_tokens=256, temperature=0.7, top_p=0.9, show_tqdm=True):
        self.append_turn("user", user_input)
        self._truncate_history()
        input_ids = self._build_inputs()
        n_tokens = self._count_tokens(input_ids)
        if show_tqdm:
            tqdm_bar = tqdm(total=self.max_context, desc="Context window usage", leave=False)
            tqdm_bar.update(n_tokens)
            tqdm_bar.close()
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Only decode the newly generated tokens
        response = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        self.append_turn("assistant", response.strip())
        return response.strip()

    def single_turn(self, user_input, **kwargs):
        self.reset()
        return self.chat(user_input, **kwargs)

if __name__ == "__main__":
    # Example usage
    chat = QwenTransformersChat()
    print("Qwen2.5-7B-Instruct Chat (Transformers)")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        response = chat.chat(user_input)
        print(f"Assistant: {response}\n") 