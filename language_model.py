import threading
import time

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
    BertTokenizerFast,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    AutoModelForPreTraining
)

from download import ModelSlugs


class LanguageModels:
    def __init__(self, download_only=False):
        self.download_only = download_only
        if download_only:
            logger.info("Pre-Downloading Language Models")

        threads = [
            threading.Thread(target=self.init_eng_qg_model),
            threading.Thread(target=self.init_cht_qg_model),
            threading.Thread(target=self.init_eng_qgg_model),
            threading.Thread(target=self.init_eng_dg_model),
            threading.Thread(target=self.init_eng_fm_model)
        ]

        start_at = time.time()

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        logger.info(f"Model loading took {(time.time() - start_at):.2f} secs")


    def init_eng_qg_model(self):
        logger.info("Start loading Enlish QG Model...")
        self.en_qg_model = AutoModelForSeq2SeqLM.from_pretrained(
            ModelSlugs.QUESTION_GENERATION_ENG_MODEL.value
        )
        self.en_qg_tokenizer = AutoTokenizer.from_pretrained(
            ModelSlugs.QUESTION_GENERATION_ENG_MODEL.value
        )
        logger.info("English QG Model loaded!")

    def init_cht_qg_model(self):
        logger.info("Start loading Chinese QG Model...")
        self.zh_qg_model = AutoModelForCausalLM.from_pretrained(
            ModelSlugs.QUESTION_GENERATION_CHT_MODEL.value
        )
        self.zh_qg_tokenizer = BertTokenizerFast.from_pretrained(
            ModelSlugs.QUESTION_GENERATION_CHT_MODEL.value
        )
        logger.info("Chinese QG Model loaded!")

    def init_eng_qgg_model(self):
        logger.info("Start loading Enlish QGG Model...")
        self.en_qgg_model = BartForConditionalGeneration.from_pretrained(
            ModelSlugs.QUESTION_GROUP_GENERATION_MODEL.value
        )
        self.en_qgg_model.to(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.en_qgg_tokenizer = BartTokenizerFast.from_pretrained(
            ModelSlugs.QUESTION_GROUP_GENERATION_MODEL.value
        )
        logger.info("English QGG Model loaded!")

    def init_eng_dg_model(self):
        logger.info("Start loading Enlish DG Model...")
        dg_models = [
            AutoModelForSeq2SeqLM.from_pretrained(
                ModelSlugs.DISTRACTOR_GENERATION_ENG_MODEL.value
            ),
            AutoModelForSeq2SeqLM.from_pretrained(
                ModelSlugs.DISTRACTOR_GENERATION_ENG_MODEL_PM.value
            ),
            AutoModelForSeq2SeqLM.from_pretrained(
                ModelSlugs.DISTRACTOR_GENERATION_ENG_MODEL_BOTH.value
            ),
        ]
        dg_tokenizers = [
            AutoTokenizer.from_pretrained(
                ModelSlugs.DISTRACTOR_GENERATION_ENG_MODEL.value
            ),
            AutoTokenizer.from_pretrained(
                ModelSlugs.DISTRACTOR_GENERATION_ENG_MODEL_PM.value
            ),
            AutoTokenizer.from_pretrained(
                ModelSlugs.DISTRACTOR_GENERATION_ENG_MODEL_BOTH.value
            ),
        ]
        rl_model = RobertaForMultipleChoice.from_pretrained(
            ModelSlugs.DISTRACTOR_GENERATION_SELECTION_RL_MODEL.value
        )
        rl_tokenizer = RobertaTokenizer.from_pretrained(
            ModelSlugs.DISTRACTOR_GENERATION_SELECTION_RL_MODEL.value
        )

        if not self.download_only:
            from distractor_generation import BartDistractorGeneration

            self.en_dis_model = BartDistractorGeneration(
                dg_models, dg_tokenizers, rl_model, rl_tokenizer
            )
        logger.info("English DG Model loaded!")


    def init_eng_fm_model(self):
        logger.info("Start loading Enlish Phishing Email Model...")
        self.en_fm_model = AutoModelForPreTraining.from_pretrained(
            ModelSlugs.PHISHING_EMAIL_GENERATION_ENG_MODEL.value
        )
        self.en_fm_tokenizer = AutoTokenizer.from_pretrained(
            'gpt2'
        )
        self.en_fm_tokenizer.add_special_tokens(
            {"bos_token": "<|BOS|>",
            "eos_token": "<|EOS|>",
            "unk_token": "<|UNK|>",
            "pad_token": "<|PAD|>",
            "sep_token": "<|SEP|>"}
        )
        logger.info("Enlish Phishing Email Model loaded!")


if __name__ == "__main__":
    models = LanguageModels(download_only=True)
