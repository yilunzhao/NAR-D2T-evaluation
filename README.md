# NAR-D2T evaluation scripts
## Requirements
- pytorch 1.4.0 (currently only test the code on pytorch-cpu version)
- huggingface transformers 2.5.1  [3.0 and above needed for moverscore]
- allennlp 0.9.0
- tensorboardX
- tqdm
- apex [optional] [might cause issues with moverscore**]
- other files required by original LogicNLG project

TODO:

# GNN
<!-- gnn
tensorflow -->

# Missing from prev
pandas
tensorboard>=1.14
rouge-score
fastDamerauLevenshtein

# Confirmed for DART
bert-score
moverscore
pyemd

# Mover score
<!-- transformers==3.1.0 -->

### Download Bleurt metrics
```
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```

### Download the NLI scorer for NLI-Acc metric
```
wget https://logicnlg.s3-us-west-2.amazonaws.com/NLI_models.zip
unzip NLI_models.zip
```

### Download the Semantic Parser for SP-Acc metric
```
wget https://logicnlg.s3-us-west-2.amazonaws.com/parser_models.zip
unzip parser_models.zip
```

## Evaluation 
Currently the following evaluation metrics are implemented:
- BLEU-1/2/3
- SP-Acc
- NLI-Acc
- ROUGE
- CO (Content Ordering, Computed with Normalized Damerau Levenshtein)
- PARENT
- METEOR
- BLEURT
- BertScore
- TER
- Moverscore



## Golden and ouput data format
The golden output should be with the same format as /reference/train_lm.json, and the model output should be in the same format as /output/GPT_gpt2_C2F_13.35.json.

## Run the evaluation script
run the evaluation script under original LogicNLG project with command:
```
python evaluation_integration.py --nli_model bert-base-multilingual-uncased --encoding gnn --nli_model_nli_model_load_from NLI_models/model_ep4.pt --fp16 --parse_model_load_from parser_models/model.pt --verify_file outputs/GPT_gpt2_C2F_13.35.json --verify_linking data/test_lm.n
```