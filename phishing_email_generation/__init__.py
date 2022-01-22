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
    title = item.title  # 起始句
    types = "Normal"
    category = item.category
    if category not in ['CRIME' ,'ENTERTAINMENT', 'POLITICS', 'SPORTS', 'BUSINESS', 'TECH', 'EDUCATION', 'HEALTHY LIVING', 'MONEY', 'CULTURE & ARTS']:
        category = 'TECH'
    formats = "Email"  # 文體

    prompt = (
                SPECIAL_TOKENS['bos_token'] + types + \
                SPECIAL_TOKENS['sep_token'] + category + \
                SPECIAL_TOKENS['sep_token'] + formats + \
                SPECIAL_TOKENS['sep_token'] + title + \
                SPECIAL_TOKENS['sep_token'] + kw
    )

    pre_len = len(types)+len(category)+len(formats)+len(title)+len(kw)
    generated = tokenizer.encode(prompt, return_tensors="pt")

    sample_outputs = model.generate(
        generated.to(model.device),
        do_sample=True,
        min_length=20,
        max_length=200,
        top_k=10,
        top_p=0.5,
        repetition_penalty=2.0,
        num_return_sequences=1, ## 只生一句
    )

    predt_email = []
    for i, sample_output in enumerate(sample_outputs):
        predit_text = tokenizer.decode(sample_output, skip_special_tokens=True)[pre_len:]
        predt_email.append(predit_text)

    return predt_email
