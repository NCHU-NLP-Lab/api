from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Question(BaseModel):
    data: str


class Answer(BaseModel):
    tag: str
    start_at: int
    end_at: int


class QuestionAndAnswer(BaseModel):
    tag: str
    start_at: int
    end_at: int
    questions: List[str] = []


class QGItem(BaseModel):
    article: str
    answer: Answer


class DisItem(BaseModel):
    article: str
    answer: Answer
    question: str
    gen_quantity: int


class DistractorSelectionStrategry(str, Enum):
    RL = "RL"
    GA = "GA"


class GenerationOrder(BaseModel):
    context: str
    question_group_size: Optional[int] = 5
    candidate_pool_size: Optional[int] = 10


class DistractorGroupQuestionAndAnswer(BaseModel):
    question: str
    answer: str


class DistractorOrder(BaseModel):
    context: str
    question_and_answers: List[DistractorGroupQuestionAndAnswer]


class Distractors(BaseModel):
    distractors: List[str]


#
# Export
#


class ExportOption(BaseModel):
    text: str
    is_answer: bool


class ExportQuestionPair(BaseModel):
    question: str
    options: List[ExportOption]


class ExportSet(BaseModel):
    context: str
    question_pairs: List[ExportQuestionPair]


#
# phishing-email
#


class FMGItem(BaseModel):
    keywords: List[str]
    category: str
    title: str
    
