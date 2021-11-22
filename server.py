import json
import os
from typing import List

from fastapi import BackgroundTasks, Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from data.model import (
    DisItem,
    DistractorOrder,
    Distractors,
    DistractorSelectionStrategry,
    ExportSet,
    FMGItem,
    GenerationOrder,
    QGItem,
)
from distractor_group_generation import generate as generate_dgg_en_us
from language_model import LanguageModels
from phishing_email_generation import generate as generate_fm_en_us
from question_generation.en_us import generate as generate_qg_en_us
from question_generation.zh_tw import generate as generate_qg_zh_tw
from question_group_generation import generate as generate_qgg_en_us
from utils import delete_later, export_file, load_examples

# Initialize Language Models
models = LanguageModels()

# Initialize example data
examples = load_examples()

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
    format: str,
    background_tasks: BackgroundTasks,
    qa_pairs: List[ExportSet] = Body(None, examples=examples.get("export-qa-pairs")),
):
    file_path = export_file(qa_pairs, format)
    background_tasks.add_task(delete_later, file_path)
    return FileResponse(
        file_path,
        filename=file_path.name,
        headers={"Access-Control-Expose-Headers": "Content-Disposition"},
    )


@app.post("/en-US/generate-question")
async def generate_en_question(
    item: QGItem = Body(None, examples=examples.get("generate-question/en-US")),
):
    return generate_qg_en_us(
        model=models.en_qg_model, tokenizer=models.en_qg_tokenizer, item=item
    )


@app.post("/en-US/generate-distractor")
async def generate_en_distractor(
    item: DisItem = Body(None, examples=examples.get("generate-distractor/en-US")),
    strategy: DistractorSelectionStrategry = DistractorSelectionStrategry.RL,
):
    if strategy is DistractorSelectionStrategry.RL:
        decodes = models.en_dis_model.generate_distractor(
            item.article,
            item.question,
            json.dumps(item.answer.dict()),
            item.gen_quantity,
            strategy,
        )
        return Distractors(distractors=decodes)
    elif strategy is DistractorSelectionStrategry.GA:
        return Distractors()


@app.post("/en-US/generate-group-distractor")
async def generate_en_group_distractors(
    order: DistractorOrder = Body(
        None, examples=examples.get("generate-group-distractor/en-US")
    ),
):
    return generate_dgg_en_us(model=models.en_dis_model, order=order)


@app.post("/zh-TW/generate-question")
async def generate_zh_question(
    item: QGItem = Body(None, examples=examples.get("generate-question/zh-TW")),
):
    return generate_qg_zh_tw(
        model=models.zh_qg_model, tokenizer=models.zh_qg_tokenizer, item=item
    )


@app.post("/zh-TW/generate-distractor")
async def generate_zh_distractor(item: DisItem):
    # TODO: Implement this
    pass


@app.post("/en-US/generate-question-group")
async def generate(
    order: GenerationOrder = Body(
        None, examples=examples.get("generate-question-group/en-US")
    ),
):
    return generate_qgg_en_us(
        model=models.en_qgg_model, tokenizer=models.en_qgg_tokenizer, order=order
    )


@app.post("/en-US/generate-phishing-email")
async def generate_en_phishing_email(item: FMGItem):
    return generate_fm_en_us(
        model=models.en_fm_model, tokenizer=models.en_fm_tokenizer, item=item
    )
