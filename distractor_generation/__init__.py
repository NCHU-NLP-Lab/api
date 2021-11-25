import itertools as it
import json
import os
from functools import lru_cache

import torch
from loguru import logger
from nlgeval import NLGEval
from torch.distributions import Categorical
from nlp2 import *

from config import max_length
from question_group_generation.optimizer import GAOptimizer


def prepare_dis_model_ga_input_ids(article, question, answer, tokenizer):
    context_input = tokenizer(
        article, max_length=max_length - 52, add_special_tokens=True, truncation=True
    )
    question_input = tokenizer(
        question, max_length=20, add_special_tokens=True, truncation=True
    )
    answer_input = tokenizer(
        answer, max_length=16, add_special_tokens=True, truncation=True
    )

    input_ids = (
        context_input["input_ids"]
        + question_input["input_ids"][1:]
        + answer_input["input_ids"][1:]
    )
    return torch.LongTensor([input_ids])


def prepare_dis_model_input_ids(
    article, question, answer, ans_start, ans_end, tokenizer
):
    sep_token_id = tokenizer.sep_token_id
    article_max_length = max_length - 52  # 後面會手動插入2個sep_token
    article_max_length -= 20  # 預留20個token空間
    article_input = tokenizer(
        tokenizer.cls_token + article, add_special_tokens=False, return_length=True
    )
    # logger.debug(article_input)
    article_length = article_input["length"]
    if type(article_length) is list:
        article_length = article_length[0]

    # 當文章過長，依據答案位置重新裁切文章
    if article_length > article_max_length:
        slice_length = int(article_max_length / 2)
        mid_index = int((ans_start + ans_end) / 2)
        new_input_ids = article_input["input_ids"][
            mid_index - slice_length : mid_index + slice_length
        ]
        article_input["input_ids"] = new_input_ids

    question_input = tokenizer(
        question,
        max_length=30,
        return_length=True,
        add_special_tokens=False,
        truncation=True,
    )

    answer_input = tokenizer(
        answer,
        max_length=20,
        return_length=True,
        add_special_tokens=False,
        truncation=True,
    )

    final_input_ids = (
        article_input["input_ids"]
        + [sep_token_id]
        + question_input["input_ids"]
        + [sep_token_id]
        + answer_input["input_ids"]
    )
    total_legnth = len(final_input_ids)
    assert total_legnth <= max_length
    return torch.LongTensor([final_input_ids]), total_legnth


class BartDistractorGeneration:
    def __init__(
        self, dg_models, dg_tokenizer, dg_selection_models, dg_selection_tokenizer
    ):
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

        self.dg_models = dg_models

        self.dg_tokenizers = dg_tokenizer

        for dg_model in self.dg_models:
            dg_model.to(os.environ.get("BDG_DEVICE", "cpu"))

        self.model = dg_selection_models
        self.model.eval()
        self.model.to(os.environ.get("BDG_CLF_DEVICE", "cpu"))
        self.tokenizer = dg_selection_tokenizer

    @lru_cache(maxsize=1000)
    def generate_distractor(self, context, question, answer, gen_quantity, strategy):
        if type(answer) is str:
            from data.model import Answer

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

        if strategy == "RL":
            return self._selection_with_rl(
                context, question, answer.tag, all_options, gen_quantity
            )
        elif strategy == "GA":
            return self._selection_with_ga(
                context, question, answer.tag, all_options, gen_quantity
            )

    def _selection_with_rl(self, context, question, answer, all_options, gen_quantity):
        max_combin = [0, []]
        for combin in set(it.combinations(all_options, gen_quantity)):
            options = list(combin) + [answer]
            keep = True
            for i in set(it.combinations(options, 2)):
                a = " ".join([char for char in split_lines_by_punc([i[0]])])
                b = " ".join([char for char in split_lines_by_punc([i[1]])])
                metrics_dict = self.nlgeval.compute_individual_metrics([a], b)
                if metrics_dict["Bleu_1"] > 0.40:
                    keep = False
                    break
            if keep:
                from config import max_length

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

    @lru_cache(maxsize=1000)
    def generate_distractor_ga(self, context, question, answer, gen_quantity):
        if answer == "":
            logger.warning("answer is null")
            return []
        all_options = []
        for i, (dg_tokenizer, dg_model) in enumerate(
            zip(self.dg_tokenizers, self.dg_models)
        ):
            d_input_ids = prepare_dis_model_ga_input_ids(
                context, question, answer, dg_tokenizer
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
                # logger.info(f"{i} {option}")
                all_options.append(option)
        # logger.info(all_options)
        # return all_options
        return self._selection_with_ga(
            context, question, answer, all_options, gen_quantity
        )

    def _selection_with_ga(self, context, question, answer, all_options, gen_quantity):
        ga_optim = GAOptimizer(len(all_options), gen_quantity)
        return ga_optim.optimize(all_options, context)[:gen_quantity]
