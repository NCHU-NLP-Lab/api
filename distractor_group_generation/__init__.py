from config import max_length
from data_model import DistractorOrder
from loguru import logger
from question_group_generation.scorer import CoverageScorer
from transformers import AutoModel


def generate(model: AutoModel, order: DistractorOrder):
    MAX_CONTEXT_LENGTH = max_length - 52
    tokenizer = model.dg_tokenizers[0]
    tokenize_result = tokenizer.batch_encode_plus(
        [order.context],
        stride=MAX_CONTEXT_LENGTH - int(MAX_CONTEXT_LENGTH * 0.7),
        max_length=MAX_CONTEXT_LENGTH,
        truncation=True,
        add_special_tokens=False,
        return_overflowing_tokens=True,
        return_length=True,
    )
    # logger.debug(tokenize_result)

    # 由於內文有長度限制；計算問句最匹配的內文段落
    keyword_coverage_scorer = CoverageScorer()
    cqas = []
    for question_and_answer in order.question_and_answers:
        question = question_and_answer.question
        answer = question_and_answer.answer
        score = 0.0
        paragraph = tokenizer.decode(tokenize_result.input_ids[0])
        for input_ids in tokenize_result.input_ids:
            _paragraph = tokenizer.decode(input_ids)
            _score = keyword_coverage_scorer._compute_coverage_score(
                [question], _paragraph
            )
            # logger.debug(f"Q:{question} A:{answer} score:{score}")

            if _score > score:
                score = _score
                paragraph = _paragraph

        cqas.append({"context": paragraph, "question": question, "answer": answer})

    outs = []
    for cqa in cqas:
        options = model.generate_distractor_ga(
            context=cqa["context"],
            question=cqa["question"],
            answer=cqa["answer"],
            gen_quantity=3,
        )
        logger.info(f"Q:{cqa['question']} A:{cqa['answer']} O:{options}")
        outs.append(
            {
                "_context": cqa["context"],
                "options": options,
                "question": cqa["question"],
                "answer": cqa["answer"],
            }
        )
    return {"distractors": outs}
