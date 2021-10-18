import torch
from loguru import logger

from config import hl_token, max_length


def prepare_qg_model_input_ids(article, start_at, end_at, tokenizer):
    hl_context = f"{article[:start_at]}{hl_token}{article[start_at:end_at]}{hl_token}{article[end_at:]}"
    logger.info(hl_context)
    model_input = tokenizer(hl_context, return_length=True)

    input_length = model_input["length"][0]
    if input_length > max_length:
        hl_token_id = tokenizer.convert_tokens_to_ids([hl_token])[0]
        slice_length = int(max_length / 2)
        mid_index = model_input["input_ids"].index(hl_token_id)
        new_input_ids = model_input["input_ids"][
            mid_index - slice_length : mid_index + slice_length
        ]
        model_input["input_ids"] = new_input_ids
    return torch.LongTensor([model_input["input_ids"]]), input_length
