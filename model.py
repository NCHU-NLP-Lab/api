from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class QuestionAndAnswer(BaseModel):
    tag: str
    start_at: int
    end_at: int
    questions: List[str] = []


class Answer(BaseModel):
    tag: str
    start_at: int
    end_at: int


class QGItem(BaseModel):
    article: str
    answer: Answer


class EnQGItem(QGItem):
    class Config:
        schema_extra = {
            "example": {
                "article": "Harry Potter is a series of seven fantasy novels written by British author, J. K. Rowling.",
                "answer": {"tag": "J. K. Rowling", "start_at": 76, "end_at": 88},
            }
        }


class ZhQGItem(QGItem):
    class Config:
        schema_extra = {
            "example": {
                "article": "英國作家J·K·羅琳的兒童奇幻文學系列小說，描寫主角哈利波特在霍格華茲魔法學校7年學習生活中的冒險故事；該系列被翻譯成75種語言",
                "answer": {"tag": "冒險故事", "start_at": 47, "end_at": 50},
            }
        }


class DisItem(BaseModel):
    article: str
    answer: Answer
    question: str
    gen_quantity: int


class EnDisItem(DisItem):
    class Config:
        schema_extra = {
            "example": {
                "article": "Harry Potter is a series of seven fantasy novels written by British author, J. K. Rowling.",
                "answer": {"tag": "J. K. Rowling", "start_at": 76, "end_at": 88},
                "question": "Who wrote Harry Potter?",
                "gen_quantity": 3,
            }
        }


class ZhDisItem(DisItem):
    class Config:
        schema_extra = {
            "example": {
                "article": "英國作家J·K·羅琳的兒童奇幻文學系列小說，描寫主角哈利波特在霍格華茲魔法學校7年學習生活中的冒險故事；該系列被翻譯成75種語言",
                "answer": {"tag": "冒險故事", "start_at": 47, "end_at": 50},
                "question": "哈利波特是一本怎麼樣的小說?",
                "gen_quantity": 3,
            }
        }


class DistractorSelectionStrategry(str, Enum):
    RL = "RL"
    GA = "GA"



class GenerationOrder(BaseModel):
    context: str
    question_group_size: Optional[int] = 5
    candidate_pool_size: Optional[int] = 10

    class Config:
        schema_extra = {
            "example": {
                "context": "Facebook is an American online social media and social networking service based in Menlo Park, California, and a flagship service of the namesake company Facebook, Inc. It was founded by Mark Zuckerberg, along with fellow Harvard College students and roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, and Chris Hughes. The founders of Facebook initially limited membership to Harvard students. Membership was expanded to Columbia, Stanford, and Yale before being expanded to the rest of the Ivy League, MIT, NYU, Boston University, then various other universities in the United States and Canada, and lastly high school students. Since 2006, anyone who claims to be at least 13 years old has been allowed to become a registered user of Facebook, though this may vary depending on local laws. The name comes from the face book directories often given to American university students."
            }
        }


class Distractors(BaseModel):
    distractors: List[str]


class ExportOption(BaseModel):
    text: str
    is_answer: bool


class ExportQuestionPair(BaseModel):
    question: str
    options: List[ExportOption]


class ExportSet(BaseModel):
    context: str
    question_pairs: List[ExportQuestionPair]

    class Config:
        schema_extra = {
            "example": {
                "context": 'Humanity needs to "grow up" and deal with the issue of climate change, British Prime Minister Boris Johnson told world leaders at the United Nations General Assembly in New York on Wednesday. Johnson, a last-minute addition to the speakers\' list that day, slammed the world\'s inadequate response to the climate crisis and urged humanity to "listen to the warnings of the scientists," pointing to the Covid-19 pandemic as "an example of gloomy scientists being proved right."',
                "question_pairs": [
                    {
                        "question": "Who is the prime minister of United Kingdom?",
                        "options": [
                            {
                                "text": "The United Nations General Nations president",
                                "is_answer": False,
                            },
                            {
                                "text": "British Prime Ministeroris Johnson",
                                "is_answer": False,
                            },
                            {"text": "Boris Johnson", "is_answer": True},
                            {"text": "Boris Johnson's father.", "is_answer": False},
                        ],
                    }
                ],
            }
        }
