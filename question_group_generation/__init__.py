import re

import torch
from config import max_length
from data_model import GenerationOrder
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from .optimizer import GAOptimizer


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


def generate(model: AutoModel, tokenizer: AutoTokenizer, order: GenerationOrder):
    context = order.context
    question_group_size = order.question_group_size
    candidate_pool_size = order.candidate_pool_size

    #
    if candidate_pool_size < question_group_size:
        return {
            "message": "`candidate_pool_size` must bigger than `question_group_size`"
        }, 400
    if candidate_pool_size > 20:
        return {"message": "`candidate_pool_size` must smaller than 20"}, 400
    if question_group_size > 10:
        return {"message": "`question_group_size` must smaller than 10"}, 400

    tokenize_result = tokenizer.batch_encode_plus(
        [context],
        stride=max_length - int(max_length * 0.7),
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_overflowing_tokens=True,
        return_length=True,
    )
    candidate_questions = []
    logger.info(f"Size of tokenize_result.input_ids:{len(tokenize_result.input_ids)}")

    if len(tokenize_result.input_ids) >= 10:
        logger.warning(
            f"Force cut tokenize_result.input_ids({len(tokenize_result.input_ids)}) to 10, it's too big"
        )
        tokenize_result.input_ids = tokenize_result.input_ids[:10]

    for input_ids in tokenize_result.input_ids:
        candidate_questions += feedback_generation(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            feedback_times=order.candidate_pool_size,
        )
    logger.info(f"Size of candidate_questions:{len(candidate_questions)}")

    while len(candidate_questions) > question_group_size:
        qgg_optim = GAOptimizer(len(candidate_questions), question_group_size)
        candidate_questions = qgg_optim.optimize(candidate_questions, context)
    return {"question_group": candidate_questions}
