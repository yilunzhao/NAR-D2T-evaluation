import json
from NLI_models import *
import os
import statistics
import subprocess
import tempfile
import pandas
from transformers import BertTokenizer
import torch
import torch.optim as optim
import scipy.stats as ss
import argparse
from itertools import chain, product
import sys
import random
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

# add for bleu
import nltk
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# add for SP-Acc
from datetime import datetime
from APIs import all_funcs
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from Model import BERTRanker
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from multiprocessing import Pool
import multiprocessing
from parser import Parser, recursion, program_eq, split_prog
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

# add for ROUGE
from rouge_score import rouge_scorer

# add for PARENT
from table_text_eval import _table_reader_v2, parent
from parse_table import parse_table_logicnlg_webnlg, parse_table_totto_webnlg

# add for CO (Content Ordering)
from fastDamerauLevenshtein import damerauLevenshtein

# add for DART metrics
from moverscore_v2 import get_idf_dict, word_mover_score
import bleurt.score
import bert_score

# import gnn.GNN as GNN
# from NLI_models import GNN
# TODO: close files

# environment: pytorch==1.4.0 (cpu-only), transformers==2.8.1, allennlp==0.9.0

# command to run the evaluation scripts:
# python evaluation_integration.py --model bert-base-multilingual-uncased --encoding gnn --load_from NLI_models/model_ep4.pt --fp16 --parse_model_load_from parser_models/model.pt --verify_file outputs/GPT_gpt2_C2F_13.35.json --verify_linking data/test_lm.json
# device = torch.device('cuda')
device = torch.device("cpu")


def parse_opt():
    # general
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify_file", default=None, type=str, help="Input verify file"
    )
    parser.add_argument(
        "--verify_linking",
        default=None,
        type=str,
        help="Link file to obtain meta information",
    )
    parser.add_argument("--batch_size", default=16, type=int, help="Max gradient norm.")
    parser.add_argument(
        "--csv_path", default="data/all_csv", type=str, help="all_csv path"
    )

    # specifically for NLI-Acc
    parser.add_argument(
        "--dim", default=768, type=int, help="dimentionality of the model"
    )
    parser.add_argument(
        "--head", default=4, type=int, help="how many heads in the each self-attention"
    )
    parser.add_argument(
        "--layers", default=3, type=int, help="how many layers in self-attention"
    )
    parser.add_argument("--max_len", default=30, type=int, help="maximum length")
    parser.add_argument(
        "--fp16", default=False, action="store_true", help="whether to use fp16"
    )
    parser.add_argument("--lr_default", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "--nli_model_nli_model_load_from",
        default="",
        type=str,
        help="whether to train or test the model",
    )
    parser.add_argument(
        "--nli_model",
        default="bert-base-multilingual-uncased",
        type=str,
        help="base model",
    )

    parser.add_argument(
        "--encoding",
        default="concat",
        type=str,
        help="the type of table encoder; choose from concat|row|cell",
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="sequence length constraint"
    )
    parser.add_argument("--max_batch_size", default=12, type=int, help="batch size")
    parser.add_argument(
        "--attention",
        default="cross",
        type=str,
        help="the attention used for interaction between statement and table",
    )

    # specifically for SP-Acc
    parser.add_argument(
        "--parse_model_type", default="bert", type=str, help="the model type"
    )
    parser.add_argument(
        "--parse_model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="number of workers in the dataloader",
    )
    parser.add_argument(
        "--parse_model_load_from",
        default=None,
        type=str,
        help="which model to load from",
    )
    parser.add_argument(
        "--cache_dir", default="/tmp/", type=str, help="where to cache the BERT model"
    )

    # specifically for PARENT
    parser.add_argument(
        "--verify_table",
        default=None,
        type=str,
        help="For LogicNLG: Path to folder containing CSVs of tables, For ToTTo: Path to textfile containing one table per row",
    )
    parser.add_argument(
        "--table_type",
        default="totto",
        type=str,
        help="table format, either 'totto' or 'logicnlg'",
    )

    args = parser.parse_args()
    return args


# BLEU_1/2/3
def calculate_bleu(args):
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    def func_compute_bleu(table_id, reference, hypothesis):
        sent_bleus_1, sent_bleus_2, sent_bleus_3 = [], [], []
        reference = [_[0].lower().split(" ") for _ in reference[table_id]]
        for hyp in hypothesis[table_id]:
            hyps = hyp.lower().split()
            sent_bleus_1.append(
                nltk.translate.bleu_score.sentence_bleu(
                    reference, hyps, weights=(1, 0, 0)
                )
            )
            sent_bleus_2.append(
                nltk.translate.bleu_score.sentence_bleu(
                    reference, hyps, weights=(0.5, 0.5, 0)
                )
            )
            sent_bleus_3.append(
                nltk.translate.bleu_score.sentence_bleu(
                    reference, hyps, weights=(0.33, 0.33, 0.33)
                )
            )
        return sent_bleus_1, sent_bleus_2, sent_bleus_3

    sent_bleus_1, sent_bleus_2, sent_bleus_3 = [], [], []
    for table_id in hypothesis.keys():
        cur_sent_bleus_1, cur_sent_bleus_2, cur_sent_bleus_3 = func_compute_bleu(
            table_id, reference, hypothesis
        )

        sent_bleus_1.extend(cur_sent_bleus_1)
        sent_bleus_2.extend(cur_sent_bleus_2)
        sent_bleus_3.extend(cur_sent_bleus_3)

    bleu_1 = sum(sent_bleus_1) / len(sent_bleus_1)
    bleu_2 = sum(sent_bleus_2) / len(sent_bleus_2)
    bleu_3 = sum(sent_bleus_3) / len(sent_bleus_3)

    print("bleu_1: {} / bleu_2: {} / bleu_3: {}".format(bleu_1, bleu_2, bleu_3))


# ROUGE
def calculate_rouge(args):

    # Instantiate rouge scorer from rouge_scorer package
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    def func_compute_rouge(table_id, reference, hypothesis, scorer=scorer):

        # Generate list of reference and predicted sentences
        sent_rouge_1, sent_rouge_2, sent_rouge_L = [], [], []
        references = [_[0].lower() for _ in reference[table_id]]
        hypotheses = [_.lower() for _ in hypothesis[table_id]]

        # Iterate through all pairs of <reference>-<hypothesis> for this table
        for (ref, hyp) in list(product(references, hypotheses)):
            output = scorer.score(ref, hyp)
            sent_rouge_1.append(output["rouge1"].fmeasure)
            sent_rouge_2.append(output["rouge2"].fmeasure)
            sent_rouge_L.append(output["rougeL"].fmeasure)

        return sent_rouge_1, sent_rouge_2, sent_rouge_L

    sent_rouge_1, sent_rouge_2, sent_rouge_L = [], [], []
    for table_id in hypothesis.keys():
        cur_sent_rouge_1, cur_sent_rouge_2, cur_sent_rouge_L = func_compute_rouge(
            table_id, reference, hypothesis
        )
        sent_rouge_1.extend(cur_sent_rouge_1)
        sent_rouge_2.extend(cur_sent_rouge_2)
        sent_rouge_L.extend(cur_sent_rouge_L)

    rouge_1 = sum(sent_rouge_1) / len(sent_rouge_1)
    rouge_2 = sum(sent_rouge_2) / len(sent_rouge_2)
    rouge_L = sum(sent_rouge_L) / len(sent_rouge_L)

    print("rouge_1: {} / rouge_2: {} / rouge_L: {}".format(rouge_1, rouge_2, rouge_L))


# CO (Content Ordering, Computed with Normalized Damerau Levenshtein)
def calculate_co(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    def func_compute_co(table_id, reference, hypothesis):

        # Generate list of reference and predicted sentences
        sent_co = []
        references = [_[0].lower() for _ in reference[table_id]]
        hypotheses = [_.lower() for _ in hypothesis[table_id]]

        # Iterate through all pairs of <reference>-<hypothesis> for this table
        for (ref, hyp) in list(product(references, hypotheses)):
            sent_co.append(1 - damerauLevenshtein(ref, hyp, similarity=True))

        return sent_co

    sent_co = []
    for table_id in hypothesis.keys():
        cur_sent_co = func_compute_co(table_id, reference, hypothesis)
        sent_co.extend(cur_sent_co)

    co = sum(sent_co) / len(sent_co)

    print(
        "CO (Content Ordering, AKA Normalized Damerau Levenshtein Distance): {}".format(
            co
        )
    )


# PARENT
def calculate_parent(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)
    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)
    # Read in tables
    if args.table_type == "totto":
        pass
    elif args.table_type == "logicnlg":
        table = parse_table_logicnlg_webnlg(args.verify_table)
    else:
        return None

    assert len(hypothesis) == len(reference)

    def func_compute_parent(table_id, reference, hypothesis, table):

        sent_parent = []

        # Generate references for a given table as a list of lists
        references = [[_[0].lower().split(" ") for _ in reference[table_id]]]

        # Loop through each predicted statement for the table
        for hyp in hypothesis[table_id]:
            predictions = [
                hyp.lower().split()
            ]  # Format predicted sentence as a list of tokens
            tables = _table_reader_v2(
                table[table_id]
            )  # Format table as a generator using _table_reader_v2
            parent_scores = parent(
                predictions=predictions, references=references, tables=tables
            )
            sent_parent.append(
                parent_scores[2]
            )  # Take second index, which corresponds to PARENT score
        return sent_parent

    sent_parent = []
    for table_id in hypothesis.keys():
        cur_sent_parent = func_compute_parent(table_id, reference, hypothesis, table)
        sent_parent.extend(cur_sent_parent)

    parent_score = sum(sent_parent) / len(sent_parent)
    print("parent: {}".format(parent_score))


# NLI-Acc
def nli_forward_pass(f, example, model, args):
    table = pandas.read_csv("{}/{}".format(args.csv_path, f), "#")
    table = table.head(40)

    cols = table.columns

    statements = example[0]
    sub_cols = example[1]
    labels = example[2]
    title = example[3]

    tab_len = len(table)
    batch_size = len(statements)
    if "gnn" in args.encoding:
        texts = []
        segs = []
        masks = []
        mapping = {}
        cur_index = 0
        for sub_col, stat in zip(sub_cols, statements):
            table_inp = []
            lengths = []

            stat_inp = tokenizer.tokenize("[CLS] " + stat + " [SEP]")
            tit_inp = tokenizer.tokenize("title is : " + title + " .")
            mapping[(cur_index, -1, -1)] = (0, len(stat_inp))

            prev_position = position = len(stat_inp) + len(tit_inp)
            for i in range(len(table)):
                tmp = tokenizer.tokenize("row {} is : ".format(i + 1))
                table_inp.extend(tmp)
                position += len(tmp)

                entry = table.iloc[i]
                for j, col in enumerate(sub_col):
                    tmp = tokenizer.tokenize(
                        "{} is {} , ".format(cols[col], entry[col])
                    )
                    mapping[(cur_index, i, j)] = (position, position + len(tmp))
                    table_inp.extend(tmp)
                    position += len(tmp)

                lengths.append(position - prev_position)
                prev_position = position

            # Tokens
            tokens = stat_inp + tit_inp + table_inp
            tokens = tokens[: args.max_length]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            texts.append(token_ids)

            # Segment Ids
            seg = [0] * len(stat_inp) + [1] * (len(tit_inp) + len(table_inp))
            seg = seg[: args.max_length]
            segs.append(seg)

            # Masks
            mask = torch.zeros(len(token_ids), len(token_ids))
            start = 0
            if args.encoding == "gnn_ind":
                mask[start : start + len(stat_inp), start : start + len(stat_inp)] = 1
            else:
                mask[start : start + len(stat_inp), :] = 1
            start += len(stat_inp)

            mask[start : start + len(tit_inp), start : start + len(tit_inp)] = 1

            start += len(tit_inp)
            for l in lengths:
                if args.encoding != "gnn_ind":
                    mask[start : start + l, : len(stat_inp) + len(tit_inp)] = 1
                mask[start : start + l, start : start + l] = 1
                start += l
            masks.append(mask)
            cur_index += 1

        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            # Padding the mask
            tmp = torch.zeros(max_len, max_len)
            tmp[: masks[i].shape[0], : masks[i].shape[1]] = masks[i]
            masks[i] = tmp.unsqueeze(0)

            # Padding the Segmentation
            segs[i] = segs[i] + [0] * (max_len - len(segs[i]))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        # Transform into tensor vectors
        inps = torch.tensor(texts).to(device)
        seg_inps = torch.tensor(segs).to(device)
        mask_inps = torch.cat(masks, 0).to(device)

        inputs = {
            "input_ids": inps,
            "attention_mask": mask_inps,
            "token_type_ids": seg_inps,
        }
        representation = model("row", **inputs)[0]

        max_len_col = max([len(_) for _ in sub_cols])
        max_len_stat = max([mapping[(_, -1, -1)][1] for _ in range(batch_size)])
        stat_representation = torch.zeros(
            batch_size, max_len_stat, representation.shape[-1]
        )
        graph_representation = torch.zeros(
            batch_size, tab_len, max_len_col, representation.shape[-1]
        )

        table_masks = []
        stat_masks = []
        for i in range(batch_size):
            mask = []
            for j in range(tab_len):
                for k in range(max_len_col):
                    if (i, j, k) in mapping:
                        start, end = mapping[(i, j, k)]
                        if start < representation.shape[1]:
                            tmp = representation[i, start:end]
                            tmp = torch.mean(tmp, 0)
                            graph_representation[i][j][k] = tmp
                            mask.append(1)
                        else:
                            mask.append(0)
                    else:
                        mask.append(0)
            table_masks.append(mask)

            start, end = mapping[(i, -1, -1)]
            stat_representation[i, start:end] = representation[i, start:end]
            stat_masks.append([1] * end + [0] * (max_len_stat - end))

        stat_representation = stat_representation.to(device)
        graph_representation = graph_representation.view(
            batch_size, -1, graph_representation.shape[-1]
        ).to(device)

        if args.attention == "self":
            x_masks = torch.cat(
                [torch.tensor(stat_masks), torch.tensor(table_masks)], 1
            ).to(device)
            representation = torch.cat([stat_representation, graph_representation], 1)
            inputs = {
                "x": representation.to(device),
                "x_mask": (1 - x_masks).unsqueeze(1).unsqueeze(2).bool(),
            }
            logits = model("sa", **inputs)
        elif args.attention == "cross":
            inputs = {
                "x": stat_representation,
                "x_mask": torch.tensor(stat_masks).to(device),
                "y": graph_representation,
                "y_mask": torch.tensor(table_masks).to(device),
            }
            logits = model("sa", **inputs)

    else:
        raise NotImplementedError

    labels = torch.LongTensor(labels).to(device)

    return logits, labels


def calculate_nli_metric(args):
    model = GNN(
        args.dim,
        args.head,
        args.nli_model,
        config,
        2,
        layers=args.layers,
        attention=args.attention,
    )
    model.to(device)

    model.load_state_dict(torch.load(args.nli_model_nli_model_load_from))
    model.eval()
    with open(args.verify_file, "r") as f:
        examples = json.load(f)

    with open(args.verify_linking, "r") as f:
        linking = json.load(f)

    print("loading file from {}".format(args.verify_file))
    files = list(examples.keys())

    succ, fail = 0, 0
    with torch.no_grad():
        correct, total = 0, 0
        for f in tqdm(files, "Evaluation"):
            r = []
            cols = []
            labels = []
            title = linking[f][0][2]
            for inst, link in zip(examples[f], linking[f]):
                r.append(inst)
                cols.append(link[1])
                labels.append(-1)
            examples[f] = [r, cols, labels, title]

            logits, labels = nli_forward_pass(f, examples[f], model, args)

            preds = torch.argmax(logits, -1)

            succ += torch.sum(preds).item()
            total += preds.shape[0]

        print("the final accuracy is {}".format(succ / total))


# SP-Acc
MODEL_CLASSES = {"bert": (BertConfig, BertModel, BertTokenizer)}


class ParseNLIDataset(Dataset):
    def __init__(self, bootstrap_data, weakly_data, tokenizer):
        self.bootstrap_data = bootstrap_data
        self.weakly_data = weakly_data
        self.tokenizer = tokenizer

        # self.sent_max_len = 80
        self.max_len = 120

    @classmethod
    def convert(cls, sent, prog, title, tokenizer, max_len):
        title = "[CLS] title : {} [SEP]".format(title)
        title_ids = tokenizer.encode(title, add_special_tokens=False)
        types = [1] * len(title_ids)

        sent = "{} [SEP]".format(sent)
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        types += [0] * len(sent_ids)

        prog = "{} [SEP]".format(prog)
        prog_ids = tokenizer.encode(prog, add_special_tokens=False)
        types += [1] * len(prog_ids)

        token_ids = title_ids + sent_ids + prog_ids

        if len(types) > max_len:
            token_ids = token_ids[:max_len]
            masks = [1] * max_len
            types = types[:max_len]
        else:
            token_ids = token_ids + [tokenizer.pad_token_id] * (max_len - len(types))
            masks = [1] * len(types) + [0] * (max_len - len(types))
            types = types + [0] * (max_len - len(types))

        return token_ids, masks, types

    def __getitem__(self, index):
        if random.random() < 0.4:
            entry = random.choice(self.bootstrap_data)
        else:
            entry = random.choice(self.weakly_data)
        # entry = random.choice(self.weakly_data)

        sent = entry[0]
        prog = entry[1]
        title = entry[2]
        label = entry[3]

        token_ids, masks, types = self.convert(
            sent, prog, title, self.tokenizer, self.max_len
        )

        token_ids = np.array(token_ids, "int64")
        types = np.array(types, "int64")
        masks = np.array(masks, "int64")
        label = np.array(label, "int64")

        return token_ids, types, masks, label

    def __len__(self):
        return len(self.bootstrap_data) + len(self.weakly_data)
        # return len(self.weakly_data)


def get_model(model_type, parse_model_name_or_path, cache_dir):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(
        parse_model_name_or_path,
        cache_dir=cache_dir,
    )
    tokenizer = tokenizer_class.from_pretrained(
        parse_model_name_or_path,
        do_lower_case=True,
        cache_dir=cache_dir,
    )
    tokenizer.add_tokens(["[{}]".format(_) for _ in all_funcs])
    tokenizer.add_tokens(["all_rows"])
    model = BERTRanker(model_class, parse_model_name_or_path, config, cache_dir)
    model.base.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def convert_program(program):
    arrays, _ = split_prog(program, True)
    for i in range(len(arrays) - 1):
        if arrays[i + 1] == "{":
            arrays[i] = "[{}]".format(arrays[i])
    return " ".join(arrays)


def calculate_sp_metric(args):
    # first parse the sent into program
    with open(args.verify_file, "r") as f:
        data = json.load(f)

    parser = Parser(args.csv_path)
    table_names = []
    sents = []
    for k, vs in data.items():
        for v in vs:
            table_names.append(k)
            sents.append(v)

    cores = multiprocessing.cpu_count()
    print("Using {} cores to run on {} instances".format(cores, len(sents)))
    pool = Pool(cores)
    results = pool.map(parser.distribute_parse, zip(table_names, sents))
    pool.close()
    pool.join()

    parsed_file = "program_{}".format(args.verify_file)
    with open(parsed_file, "w") as f:
        json.dump(results, f, indent=2)

    # then compute score
    tokenizer, model = get_model(
        args.parse_model_type, args.parse_model_name_or_path, args.cache_dir
    )
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.parse_model_load_from), False)

    with open(parsed_file, "r") as f:
        results = json.load(f)

    succ, total = 0, 0
    for sent, programs, title in results:
        labels = []
        token_ids = []
        types = []
        masks = []

        if len(programs) > 0:
            for prog in programs[:36]:
                token_id, type_, mask = ParseNLIDataset.convert(
                    sent, convert_program(prog), title, tokenizer, 120
                )
                token_ids.append(token_id)
                types.append(type_)
                masks.append(mask)
                labels.append(1 if "=True" in prog else 0)

            token_ids = torch.LongTensor(token_ids).to(device)
            types = torch.LongTensor(types).to(device)
            masks = torch.LongTensor(masks).to(device)

            probs = model.prob(token_ids, types, masks)
            if len(labels) > 8:
                tmp = []
                for _ in probs.topk(3)[1].tolist():
                    tmp.append(labels[_])
                if sum(tmp) > 0:
                    pred = 1
                else:
                    pred = 0
            else:
                pred = labels[torch.argmax(probs, 0).item()]
        else:
            pred = 0

        if pred:
            succ += 1
        total += 1
        sys.stdout.write("accuracy = {} \r".format(succ / total))

    print("SP-Acc = {}".format(succ / total))


# Metrics from DART


def rem_tempfiles(openfiles):
    for file in openfiles:
        file.close()
        os.remove(file.name)


# Bleu from DART
def calculate_bleu_dart(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    openfiles = []

    hyps_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(hyps_file)

    ref_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(ref_file)

    out_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(out_file)

    def process_data(table_id, reference, hypothesis):

        processed_hyps, processed_refs = [], []

        reference = [_[0].lower() for _ in reference[table_id]]
        for i, hyp in enumerate(hypothesis[table_id]):
            hyps = hyp.lower()
            processed_hyps.append(hyps)
            processed_refs.append(reference[i])

        with open(hyps_file.name, "a") as hyp_in:
            with open(ref_file.name, "a") as ref_in:
                hyp_in.write("\n".join(processed_hyps) + "\n")
                ref_in.write("\n".join(processed_refs) + "\n")

    for table_id in hypothesis.keys():
        process_data(table_id, reference, hypothesis)

    with open(out_file.name, "w") as ouf:
        with open(hyps_file.name, "r") as hyp_in:
            ## TODO: provide multiple reference files
            proc = subprocess.Popen(
                ["./multi-bleu.perl", ref_file.name, ref_file.name, ref_file.name],
                stdout=ouf,
                stdin=hyp_in,
                text=True,
            )
            proc.wait()

    with open(out_file.name, "r") as f:
        bleu = float(f.read().strip().split()[2].replace(",", ""))

    print("bleu_dart: {:.2f}".format(bleu))
    rem_tempfiles(openfiles)


from prepare_files import prepare_files_ter

# TER from DART
def calculate_TER(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    openfiles = []

    hyps_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(hyps_file)

    ref_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(ref_file)

    out_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(out_file)

    def process_data(table_id, reference, hypothesis):

        processed_hyps, processed_refs = [], []

        reference = [_[0].lower() for _ in reference[table_id]]
        for i, hyp in enumerate(hypothesis[table_id]):
            hyps = hyp.lower()
            processed_hyps.append(hyps)
            processed_refs.append(reference[i])

        with open(hyps_file.name, "a") as hyp_in:
            with open(ref_file.name, "a") as ref_in:
                hyp_in.write("\n".join(processed_hyps) + "\n")
                ref_in.write("\n".join(processed_refs) + "\n")

    for table_id in hypothesis.keys():
        process_data(table_id, reference, hypothesis)

    prepare_files_ter(hyps_file.name, ref_file.name)

    with open(out_file.name, "w") as ouf:
        proc = subprocess.Popen(
            [
                "java",
                "-jar",
                "tercom-0.7.25/tercom.7.25.jar",
                "-h",
                "relexicalised_predictions-ter.txt",
                "-r",
                "all-notdelex-refs-ter.txt",
            ],
            stdout=ouf,
            text=True,
        )
        proc.wait()

    with open(out_file.name, "r") as f:
        ter = float(f.readlines()[-4].strip().split()[2])

    print("TER: {:.2f}".format(ter))
    rem_tempfiles(openfiles)


# METEOR from DART
def calculate_METEOR(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    openfiles = []
    hyps_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(hyps_file)

    ref_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(ref_file)

    out_file = tempfile.NamedTemporaryFile(delete=False)
    openfiles.append(out_file)

    def process_data(table_id, reference, hypothesis):

        processed_hyps, processed_refs = [], []

        reference = [_[0].lower() for _ in reference[table_id]]
        for i, hyp in enumerate(hypothesis[table_id]):
            hyps = hyp.lower()
            processed_hyps.append(hyps)
            processed_refs.append(reference[i])

        with open(hyps_file.name, "a") as hyp_in:
            with open(ref_file.name, "a") as ref_in:
                hyp_in.write("\n".join(processed_hyps) + "\n")
                ref_in.write("\n".join(processed_refs) + "\n")

    for table_id in hypothesis.keys():
        process_data(table_id, reference, hypothesis)

    prepare_files_ter(hyps_file.name, ref_file.name)

    with open(out_file.name, "w") as ouf:
        proc = subprocess.Popen(
            [
                "java",
                "-Xmx2G",
                "-jar",
                "meteor-1.5/meteor-1.5.jar",
                "relexicalised_predictions-ter.txt",
                "all-notdelex-refs-meteor.txt",
                "-l",
                "en",
                "-norm",
                "-r",
                "8",
            ],
            stdout=ouf,
            text=True,
        )
        proc.wait()

    with open(out_file.name, "r") as f:
        meteor = float(f.readlines()[-1].strip().split()[-1])

    print("METEOR: {:.2f}".format(meteor))
    rem_tempfiles(openfiles)


# BLEURT
def calculate_bleurtscore(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    checkpoint = "BLEURT-20"

    def process_data(table_id, reference, hypothesis):

        scores_all = []

        reference = [_[0].lower() for _ in reference[table_id]][0:1]
        for i, hyp in enumerate(hypothesis[table_id]):
            hyps = [hyp.lower()]
            scorer = bleurt.score.BleurtScorer(checkpoint)
            scores = scorer.score(references=reference, candidates=hyps)
            assert type(scores) == list and len(scores) == 1
            scores_all += scores

        return scores_all

    sent_co = []
    for table_id in list(hypothesis.keys())[:2]:
        bleurt_scores = process_data(table_id, reference, hypothesis)
        sent_co.extend(bleurt_scores)

    bleurtscore = sum(sent_co) / len(sent_co)
    print("BLEURT: {:.2f}".format(bleurtscore))


# Bertscore
def calculate_bertscore(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    def process_data(table_id, reference, hypothesis):

        processed_hyps, processed_refs = [], []

        reference = [_[0].lower() for _ in reference[table_id]]
        for i, hyp in enumerate(hypothesis[table_id]):
            hyps = hyp.lower()
            processed_hyps.append(hyps)
            processed_refs.append(reference)

        return processed_hyps, processed_refs

    cands_all, refs_all = [], []
    for table_id in list(hypothesis.keys())[:2]:
        cands, refs = process_data(table_id, reference, hypothesis)
        cands_all.extend(cands)
        refs_all.extend(refs)

    P, R, F1 = bert_score.score(cands_all, refs_all, lang="en")

    print("BERTScore Precision (P): {:.2f}".format(P.mean()))
    print("BERTScore Recall (R): {:.2f}".format(R.mean()))
    print("BERTScore F1: {:.2f}".format(F1.mean()))


# Moverscore from DART
def calculate_moverscore(args):

    # Read in generated predictions (hypotheses)
    with open(args.verify_file, "r") as f:
        hypothesis = json.load(f)

    # Read in generated gold standards (references)
    with open(args.verify_linking, "r") as f:
        reference = json.load(f)

    assert len(hypothesis) == len(reference)

    def process_data(table_id, reference, hypothesis):

        processed_hyps, processed_refs = [], []

        reference = [_[0].lower() for _ in reference[table_id]]
        for hyp in hypothesis[table_id]:
            hyps = hyp.lower()
            processed_hyps.append(hyps)
            processed_refs.append(reference[0])

        return processed_hyps, processed_refs

    cands_all, refs_all = [], []
    for table_id in list(hypothesis.keys()):
        cands, refs = process_data(table_id, reference, hypothesis)
        cands_all.extend(cands)
        refs_all.extend(refs)

    idf_dict_hyp = get_idf_dict(
        cands_all
    )  # idf_dict_hyp = defaultdict(lambda: 1.)#translations is a list of candidate sentences
    idf_dict_ref = get_idf_dict(
        [ref for refset in refs_all for ref in refset]
    )  # idf_dict_ref = defaultdict(lambda: 1.)#reference is a list of reference sentences

    scores = word_mover_score(
        refs_all,
        cands_all,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
    )

    print("MoverScore: {:.2f}".format(statistics.mean(scores)))


if __name__ == "__main__":
    args = parse_opt()
    print("\nStart calculating BLEU Score")
    calculate_bleu(args)

    print("\nStart calculating ROUGE Score")
    calculate_rouge(args)

    print("\nStart calculating PARENT Score")
    calculate_parent(args)

    print("\nStart calculating CO (Content Ordering) Score")
    calculate_co(args)

    print(
        "\nStart calculating NLI-ACC, this evaluation might take around 0.5 hour for CPU-only device"
    )
    config = BertConfig.from_pretrained(args.nli_model, cache_dir="tmp/")
    tokenizer = BertTokenizer.from_pretrained(args.nli_model, cache_dir="tmp/")
    calculate_nli_metric(args)

    print(
        "\nStart calculating SP-ACC, this evaluation might take around 1 hour for CPU-only device"
    )
    calculate_sp_metric(args)

    print("\nStart calculating BLEU from DART Score")
    calculate_bleu_dart(args)

    print("\nStart calculating TER Score")
    calculate_TER(args)

    print("\nStart calculating METEOR Score")
    calculate_METEOR(args)

    print("\nStart calculating moverscore")
    calculate_moverscore(args)

    print("\nStart calculating bert-score")
    calculate_bertscore(args)

    print("\nStart calculating bleurtscore")
    calculate_bleurtscore(args)
