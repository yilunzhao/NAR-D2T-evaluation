# NAR-D2T evaluation scripts
## Requirements
- pytorch 1.4.0 (currently only support cpu only)
- huggingface transformers 2.5.1
- allennlp 0.9.0
- tensorboardX
- tqdm
- apex [optional]
- other files required by original LogicNLG project

## Evaluation 
Currently the following evaluation metrics are implemented:
- BLEU-1/2/3
- SP-Acc
- NLI-Acc
  
run the evaluation script under original LogicNLG project with command:
```
python evaluation_integration.py --model bert-base-multilingual-uncased --encoding gnn --load_from NLI_models/model_ep4.pt --fp16 --parse_model_load_from parser_models/model.pt --verify_file outputs/GPT_gpt2_C2F_13.35.json --verify_linking data/test_lm.json
```