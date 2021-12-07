from typing import Tuple

from config import max_length, max_question_length
from data.model import QGItem, QuestionAndAnswer
from transformers import AutoModel, AutoTokenizer

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
        input_ids=input_ids.to(model.device),
        max_length=max_length + max_question_length,
        early_stopping=True,
        do_sample=False,
        num_beams=10,
        num_beam_groups=5,
        diversity_penalty=-10,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        eos_token_id=tokenizer.eos_token_id,
    )

    decode_questions = []
    for output in outputs:
        decode_question = tokenizer.decode(
            output[input_length:], skip_special_tokens=True
        )
        decode_question = decode_question.replace(" ", "")
        decode_questions.append(decode_question)
    return QuestionAndAnswer(
        tag=item.answer.tag,
        start_at=item.answer.start_at,
        end_at=item.answer.end_at,
        questions=decode_questions,
    )
