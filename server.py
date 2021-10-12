import json
import os
from typing import List

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger

from config import max_length, max_question_length
from language_models import LanguageModels
from model import (
    Distractors,
    DistractorSelectionStrategry,
    EnDisItem,
    EnQGItem,
    ExportSet,
    GenerationOrder,
    QuestionAndAnswer,
    ZhDisItem,
    ZhQGItem,
)
from utils import (
    GAOptimizer,
    delete_later,
    export_file,
    feedback_generation,
    prepare_qg_model_input_ids,
)

# Initialize Language Models
models = LanguageModels()

app = FastAPI(title="NCHU NLP API", description="All-in-one NLP task", version="0.1.0")

origins = os.getenv(
    "allow_origins",
    "https://app.queratorai.com https://app2.queratorai.com http://localhost:8000 http://localhost:3000",
).split()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return RedirectResponse("docs")


@app.post("/export-qa-pairs/{format}")
async def export_qa_pairs(
    qa_pairs: List[ExportSet], format: str, background_tasks: BackgroundTasks
):
    file_path = export_file(qa_pairs, format)
    background_tasks.add_task(delete_later, file_path)
    return FileResponse(
        file_path,
        filename=file_path.name,
        headers={"Access-Control-Expose-Headers": "Content-Disposition"},
    )


@app.post("/en-US/generate-question")
async def generate_en_question(item: EnQGItem):
    article = item.article
    start_at = item.answer.start_at
    end_at = item.answer.end_at + 1

    input_ids, input_length = prepare_qg_model_input_ids(
        article, start_at, end_at, models.en_qg_tokenizer
    )
    outputs = models.en_qg_model.generate(
        input_ids=input_ids,
        max_length=max_question_length,
        early_stopping=True,
        do_sample=False,
        num_beams=10,
        num_beam_groups=5,
        diversity_penalty=0.5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
    )

    decode_questions = []
    for output in outputs:
        decode_question = models.en_qg_tokenizer.decode(
            output, skip_special_tokens=True
        )
        decode_questions.append(decode_question)
    return QuestionAndAnswer(
        tag=item.answer.tag,
        start_at=item.answer.start_at,
        end_at=item.answer.end_at,
        questions=decode_questions,
    )


@app.post("/en-US/generate-distractor")
async def generate_en_distractor(
    item: EnDisItem,
    strategy: DistractorSelectionStrategry = DistractorSelectionStrategry.RL,
):
    article = item.article
    answer = item.answer
    question = item.question
    gen_quantity = item.gen_quantity
    decodes = models.en_dis_model.generate_distractor(
        article, question, json.dumps(answer.dict()), gen_quantity, strategy
    )
    return Distractors(distractors=decodes)


@app.post("/zh-TW/generate-question")
async def generate_zh_question(item: ZhQGItem):
    article = item.article
    start_at = item.answer.start_at
    end_at = item.answer.end_at + 1

    input_ids, input_length = prepare_qg_model_input_ids(
        article, start_at, end_at, models.zh_qg_tokenizer
    )
    outputs = models.zh_qg_model.generate(
        input_ids=input_ids,
        max_length=max_length + max_question_length,
        early_stopping=True,
        do_sample=False,
        num_beams=10,
        num_beam_groups=5,
        diversity_penalty=-10,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        eos_token_id=models.zh_qg_tokenizer.eos_token_id,
    )

    decode_questions = []
    for output in outputs:
        decode_question = models.zh_qg_tokenizer.decode(
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


@app.post("/zh-TW/generate-distractor")
async def generate_zh_distractor(item: ZhDisItem):
    pass


@app.post("/zh-TW/generate-question-group")
async def generate(order: GenerationOrder):
    # return {'question_group':[
    #             'Harry Potter is a series of seven fantasy novels written by   _ .',
    #             'Who is Voldemort?',
    #             'How does the story begin?'
    #         ]}

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

    tokenize_result = models.en_qgg_tokenizer.batch_encode_plus(
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
            model=models.en_qgg_model,
            tokenizer=models.en_qgg_tokenizer,
            input_ids=input_ids,
            feedback_times=order.candidate_pool_size,
        )
    logger.info(f"Size of candidate_questions:{len(candidate_questions)}")

    while len(candidate_questions) > question_group_size:
        qgg_optim = GAOptimizer(len(candidate_questions), question_group_size)
        candidate_questions = qgg_optim.optimize(candidate_questions, context)
    return {"question_group": candidate_questions}
