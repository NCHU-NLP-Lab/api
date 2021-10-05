import torch
from config import hl_token, max_length
from loguru import logger


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


def prepare_dis_model_input_ids(
    article, question, answer, ans_start, ans_end, tokenizer
):
    sep_token_id = tokenizer.sep_token_id
    article_max_length = max_length - 52  # 後面會手動插入2個sep_token
    article_max_length -= 20  # 預留20個token空間
    article_input = tokenizer(
        tokenizer.cls_token + article, add_special_tokens=False, return_length=True
    )
    # logger.debug(article_input)
    article_length = article_input["length"]
    if type(article_length) is list:
        article_length = article_length[0]

    # 當文章過長，依據答案位置重新裁切文章
    if article_length > article_max_length:
        slice_length = int(article_max_length / 2)
        mid_index = int((ans_start + ans_end) / 2)
        new_input_ids = article_input["input_ids"][
            mid_index - slice_length : mid_index + slice_length
        ]
        article_input["input_ids"] = new_input_ids

    question_input = tokenizer(
        question,
        max_length=30,
        return_length=True,
        add_special_tokens=False,
        truncation=True,
    )

    answer_input = tokenizer(
        answer,
        max_length=20,
        return_length=True,
        add_special_tokens=False,
        truncation=True,
    )

    final_input_ids = (
        article_input["input_ids"]
        + [sep_token_id]
        + question_input["input_ids"]
        + [sep_token_id]
        + answer_input["input_ids"]
    )
    total_legnth = len(final_input_ids)
    assert total_legnth <= max_length
    return torch.LongTensor([final_input_ids]), total_legnth
