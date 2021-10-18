import logging
import threading
import time
from enum import Enum

import stanza
from loguru import logger
from transformers import AutoModel, AutoTokenizer


class ModelSlugs(str, Enum):
    QUESTION_GENERATION_ENG_MODEL = "p208p2002/bart-squad-qg-hl"
    QUESTION_GENERATION_CHT_MODEL = "p208p2002/gpt2-drcd-qg-hl"
    QUESTION_GROUP_GENERATION_MODEL = "p208p2002/qmst-qgg"
    DISTRACTOR_GENERATION_ENG_MODEL = "voidful/bart-distractor-generation"
    DISTRACTOR_GENERATION_ENG_MODEL_PM = f"{DISTRACTOR_GENERATION_ENG_MODEL}-pm"
    DISTRACTOR_GENERATION_ENG_MODEL_BOTH = f"{DISTRACTOR_GENERATION_ENG_MODEL}-both"
    DISTRACTOR_GENERATION_SELECTION_RL_MODEL = "LIAMF-USP/roberta-large-finetuned-race"
    PHISHING_EMAIL_GENERATION_ENG_MODEL = "heliart/PhishingEmailGeneration"


def download_model(model_name):
    logger.info(f"Loading: {model_name}")
    AutoModel.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Complete: {model_name}")


def main():
    # Surpress partial weights not used warning
    logging.getLogger("transformers").setLevel(logging.CRITICAL)

    start_at = time.time()

    # Setup downoad threads
    threads = [
        threading.Thread(target=stanza.download, args=("en",)),
    ]
    for model in ModelSlugs:
        threads.append(threading.Thread(target=download_model, args=(model.value,)))

    # Start download
    for thread in threads:
        thread.start()

    # Wait for all to complete
    for thread in threads:
        thread.join()

    logger.info(f"Download took {(time.time() - start_at):.2f} secs")


if __name__ == "__main__":
    main()
