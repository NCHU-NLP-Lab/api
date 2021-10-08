import re

import torch
from config import max_length

from .optim import GAOptimizer

__all__ = ["feedback_generation", "GAOptimizer"]


def feedback_generation(model, tokenizer, input_ids, feedback_times=3):
    outputs = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for i in range(feedback_times):
        gened_text = tokenizer.bos_token * (len(outputs) + 1)
        gened_ids = tokenizer(gened_text, add_special_tokens=False)["input_ids"]
        input_ids = gened_ids + input_ids
        input_ids = input_ids[:max_length]

        sample_outputs = model.generate(
            input_ids=torch.LongTensor(input_ids).unsqueeze(0).to(device),
            attention_mask=torch.LongTensor([1] * len(input_ids))
            .unsqueeze(0)
            .to(device),
            max_length=50,
            early_stopping=True,
            temperature=1.0,
            do_sample=True,
            top_p=0.9,
            top_k=10,
            num_beams=1,
            no_repeat_ngram_size=5,
            num_return_sequences=1,
        )
        sample_output = sample_outputs[0]
        decode_question = tokenizer.decode(sample_output, skip_special_tokens=False)
        decode_question = re.sub(re.escape(tokenizer.pad_token), "", decode_question)
        decode_question = re.sub(re.escape(tokenizer.eos_token), "", decode_question)
        if tokenizer.bos_token is not None:
            decode_question = re.sub(
                re.escape(tokenizer.bos_token), "", decode_question
            )
        decode_question = decode_question.strip()
        decode_question = decode_question.replace("[Q:]", "")
        outputs.append(decode_question)
    return outputs
