from torch import Tensor, nn
from transformers import T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, path, max_length: int, **hf_kwargs):
        super().__init__()
        self.max_length = max_length
        self.output_key = "last_hidden_state"

        self.tokenizer = T5Tokenizer.from_pretrained(path, max_length=max_length, legacy=False)
        self.hf_module = T5EncoderModel.from_pretrained(path, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_dis = batch_encoding["input_ids"].to(self.hf_module.device)
        attention_mask = batch_encoding["attention_mask"].to(self.hf_module.device)

        outputs = self.hf_module(
            input_ids=input_dis,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return outputs[self.output_key], attention_mask

