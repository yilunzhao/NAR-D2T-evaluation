# NAR-D2T evaluation scripts
## Requirements
- pytorch 1.4.0 (currently only test the code on pytorch-cpu version)
- huggingface transformers 2.5.1
- allennlp 0.9.0
- tensorboardX
- tqdm
- apex [optional]
- other files required by original LogicNLG project

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


## Golden and ouput data format
The golden output should be with the same format as /reference/train_lm.json, and the model output should be in the same format as /output/GPT_gpt2_C2F_13.35.json.

## Run the evaluation script
run the evaluation script under original LogicNLG project with command:
```
python evaluation_integration.py --model bert-base-multilingual-uncased --encoding gnn --load_from NLI_models/model_ep4.pt --fp16 --parse_model_load_from parser_models/model.pt --verify_file outputs/GPT_gpt2_C2F_13.35.json --verify_linking data/test_lm.json
```