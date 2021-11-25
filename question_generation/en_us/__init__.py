from typing import Tuple

from transformers import AutoModel, AutoTokenizer

from config import max_question_length
from data.model import QGItem, QuestionAndAnswer

from .. import prepare_qg_model_input_ids


def setup() -> Tuple[AutoModel, AutoTokenizer]:
    pass


def generate(
    model: AutoModel, tokenizer: AutoTokenizer, item: QGItem
) -> QuestionAndAnswer:
    article = item.article
    start_at = item.answer.start_at
    end_at = item.answer.end_at + 1

    input_ids, input_length = prepare_qg_model_input_ids(
        article, start_at, end_at, tokenizer
    )
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_question_length,
        early_stopping=True,
        do_sample=False,
        num_beam_groups=5,
        diversity_penalty=0.5,
        num_beams=10,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
    )

    decode_questions = []
    for output in outputs:
        decode_question = tokenizer.decode(output, skip_special_tokens=True)
        decode_questions.append(decode_question)
    return QuestionAndAnswer(
        tag=item.answer.tag,
        start_at=item.answer.start_at,
        end_at=item.answer.end_at,
        questions=decode_questions,
    )
