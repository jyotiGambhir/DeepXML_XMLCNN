#!/usr/bin/env bash

python3 /content/DeepXML_XMLCNN/deepXML/main.py --mb=128 --ss=499 --sequence_length=20 --load_data=1  --hidden_dims=1024 --vocab_size=150 --mn=eurlex --e=5  --ds=EUR-Lex --data_conf="/content/DeepXML_XMLCNN/deepXML/configure/datasets/EUR-Lex.yaml" --model_conf="/content/DeepXML_XMLCNN/deepXML/configure/models/AttentionXML-EUR-Lex.yaml"
# python3 main.py --data-cnf /content/DeepXML/configure/datasets/EUR-Lex.yaml --model-cnf /content/DeepXML/configure/models/AttentionXML-EUR-Lex.yaml -t 1
# python3 main.py --data-cnf /content/DeepXML/configure/datasets/EUR-Lex.yaml --model-cnf /content/DeepXML/configure/models/AttentionXML-EUR-Lex.yaml -t 2
# python3 ensemble.py -p results/$2-$1 -t 3


