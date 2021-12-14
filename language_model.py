import threading
import time
from collections import namedtuple

import stanza
import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
    BertTokenizerFast,
    RobertaForMultipleChoice,
    RobertaTokenizer,
)

ModelSpec = namedtuple("Model", ["model_class", "tokenizer_class", "name", "alias"])

MODELS_SPECS = [
    ModelSpec(
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        "p208p2002/bart-squad-qg-hl",
        "qg_en",
    ),
    ModelSpec(
        AutoModelForCausalLM,
        BertTokenizerFast,
        "p208p2002/gpt2-drcd-qg-hl",
        "qg_zh",
    ),
    ModelSpec(
        BartForConditionalGeneration,
        BartTokenizerFast,
        "p208p2002/qmst-qgg",
        "qgg_en",
    ),
    ModelSpec(
        AutoModelForCausalLM,
        AutoTokenizer,
        "gpt2",
        "pplscorer",
    ),
    ModelSpec(
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        "voidful/bart-distractor-generation",
        "_dg_en",
    ),
    ModelSpec(
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        "voidful/bart-distractor-generation-pm",
        "_dg_pm",
    ),
    ModelSpec(
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        "voidful/bart-distractor-generation-both",
        "_dg_both",
    ),
    ModelSpec(
        RobertaForMultipleChoice,
        RobertaTokenizer,
        "LIAMF-USP/roberta-large-finetuned-race",
        "_dg_rl",
    ),
    ModelSpec(
        AutoModelForPreTraining,
        AutoTokenizer,
        "nlplab/PhishingEmailGeneration",
        "fm_en",
    ),
]


class LanguageModels:
    def __init__(self, download_only=False):
        threads = [
            threading.Thread(target=self._init, args=(spec, download_only))
            for spec in MODELS_SPECS
        ]

        if download_only:
            threads.append(threading.Thread(target=stanza.download, args=("en",)))

        start_at = time.time()

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        if not download_only:
            self._post_dg_load()
        logger.info(f"Model loading took {(time.time() - start_at):.2f} secs")

    def _init(self, spec: ModelSpec, download_only: bool = False):
        logger.info(f"Start loading <{spec.name}>...")
        model = spec.model_class.from_pretrained(spec.name)
        if not download_only:
            from config import CUDA_MODELS

            model.to(
                "cuda"
                if spec.name in CUDA_MODELS and torch.cuda.is_available()
                else "cpu"
            )
        tokenizer = spec.tokenizer_class.from_pretrained(spec.name)
        setattr(self, f"{spec.alias}_model", model)
        setattr(self, f"{spec.alias}_tokenizer", tokenizer)
        logger.info(f"<{spec.name}> loaded!")

    def _post_dg_load(self):
        from distractor_generation import BartDistractorGeneration

        self.dis_en_model = BartDistractorGeneration(
            dg_models=[self._dg_en_model, self._dg_pm_model, self._dg_both_model],
            dg_tokenizer=[
                self._dg_en_tokenizer,
                self._dg_pm_tokenizer,
                self._dg_both_tokenizer,
            ],
            dg_selection_models=self._dg_rl_model,
            dg_selection_tokenizer=self._dg_rl_tokenizer,
            pplscorer_model=self.pplscorer_model,
            pplscorer_tokenizer=self.pplscorer_tokenizer,
        )


if __name__ == "__main__":
    models = LanguageModels(download_only=True)
