import random

from data.model import FMGItem
from transformers import AutoModel, AutoTokenizer

SPECIAL_TOKENS = {
    "bos_token": "<|BOS|>",
    "eos_token": "<|EOS|>",
    "unk_token": "<|UNK|>",
    "pad_token": "<|PAD|>",
    "sep_token": "<|SEP|>",
}


def _join_keywords(keywords, randomize=False):
    N = len(keywords)

    # random sampling and shuffle
    if randomize:
        M = random.choice(range(N + 1))
        keywords = keywords[:M]
        random.shuffle(keywords)

    return ",".join(keywords)


def generate(model: AutoModel, tokenizer: AutoTokenizer, item: FMGItem):
    kw = _join_keywords(item.keywords)
    title = ""

    prompt = (
        SPECIAL_TOKENS["bos_token"]
        + title
        + SPECIAL_TOKENS["sep_token"]
        + kw
        + SPECIAL_TOKENS["sep_token"]
    )

    generated = tokenizer.encode(prompt, return_tensors="pt")

    sample_outputs = model.generate(
        generated.to(model.device),
        do_sample=True,
        min_length=50,
        max_length=768,
        top_k=30,
        top_p=0.7,
        temperature=0.9,
        repetition_penalty=2.0,
        num_return_sequences=1,
    )

    sentence = []
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(title) + len(",".join(item.keywords))
        sentence.append(title + text[a:])
        # print('{} {}\n\n'.format(title,sentence[i]))

    return sentence
