import itertools as it
import json
import os
from functools import lru_cache

import torch
from loguru import logger
from nlgeval import NLGEval
from torch.distributions import Categorical
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    RobertaForMultipleChoice,
    RobertaTokenizer,
)

from ..config import max_length
from ..model import Answer
from .tokenizing import prepare_dis_model_input_ids


class BartDistractorGeneration:
    def __init__(self):
        self.nlgeval = NLGEval(
            metrics_to_omit=[
                "METEOR",
                "EmbeddingAverageCosineSimilairty",
                "SkipThoughtCS",
                "VectorExtremaCosineSimilarity",
                "GreedyMatchingScore",
                "CIDEr",
            ]
        )

        #
        self.dg_models = [
            AutoModelForSeq2SeqLM.from_pretrained("voidful/bart-distractor-generation"),
            AutoModelForSeq2SeqLM.from_pretrained(
                "voidful/bart-distractor-generation-pm"
            ),
            AutoModelForSeq2SeqLM.from_pretrained(
                "voidful/bart-distractor-generation-both"
            ),
        ]

        self.dg_tokenizers = [
            AutoTokenizer.from_pretrained("voidful/bart-distractor-generation"),
            AutoTokenizer.from_pretrained("voidful/bart-distractor-generation-pm"),
            AutoTokenizer.from_pretrained("voidful/bart-distractor-generation-both"),
        ]

        for dg_model in self.dg_models:
            dg_model.to(os.environ["BDG_DEVICE"])

        #
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )
        self.model = RobertaForMultipleChoice.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )
        self.model.eval()
        self.model.to(os.environ["BDG_CLF_DEVICE"])

    @lru_cache(maxsize=1000)
    def generate_distractor(self, context, question, answer, gen_quantity):
        if type(answer) is str:
            answer = Answer.parse_obj(json.loads(answer))

        all_options = []
        for i, (dg_tokenizer, dg_model) in enumerate(
            zip(self.dg_tokenizers, self.dg_models)
        ):
            d_input_ids, _ = prepare_dis_model_input_ids(
                context,
                question,
                answer.tag,
                answer.start_at,
                answer.end_at,
                dg_tokenizer,
            )  # 如果文章過長進行重新裁切與處理
            out_ids = dg_model.generate(
                input_ids=d_input_ids.to(dg_model.device),
                num_beams=gen_quantity * 3,
                length_penalty=0.9,
                num_beam_groups=gen_quantity,
                diversity_penalty=1.0,
                num_return_sequences=gen_quantity * 2,
            )
            for out_seq_ids in out_ids:
                option = dg_tokenizer.decode(out_seq_ids, skip_special_tokens=True)
                logger.info(f"{i} {option}")
                all_options.append(option)
        # logger.info(all_options)
        return self._selection(context, question, answer.tag, all_options, gen_quantity)

    def _selection(self, context, question, answer, all_options, gen_quantity):
        max_combin = [0, []]
        for combin in set(it.combinations(all_options, gen_quantity)):
            options = list(combin) + [answer]
            keep = True
            for i in set(it.combinations(options, 2)):
                a = "".join(
                    [
                        char if char.isalpha() or char == " " else " " + char + " "
                        for char in i[0]
                    ]
                )
                b = "".join(
                    [
                        char if char.isalpha() or char == " " else " " + char + " "
                        for char in i[1]
                    ]
                )
                metrics_dict = self.nlgeval.compute_individual_metrics([a], b)
                if metrics_dict["Bleu_1"] > 0.20:
                    keep = False
                    break
            if keep:
                prompt = context + self.tokenizer.sep_token + question
                encoding_input = []
                for choice in options:
                    encoding_input.append([prompt, choice])
                encoding_input.append([prompt, answer])
                labels = torch.tensor(len(options) - 1).unsqueeze(0)
                encoding = self.tokenizer(
                    encoding_input,
                    return_tensors="pt",
                    padding=True,
                    truncation="only_first",
                    max_length=max_length,
                )
                outputs = self.model(
                    **{
                        k: v.unsqueeze(0).to(self.model.device)
                        for k, v in encoding.items()
                    },
                    labels=labels.to(self.model.device),
                )  # batch size is 1
                entropy = (
                    Categorical(probs=torch.softmax(outputs.logits, -1))
                    .entropy()
                    .tolist()[0]
                )
                if entropy >= max_combin[0]:
                    max_combin = [entropy, options]
        return max_combin[1][:-1]
