#!/usr/bin/env bash

# python3 /content/DeepXML_XMLCNN/deepXML/main.py --mb=32 --ss=499 --sequence_length=20 --load_data=1  --hidden_dims=3801 --vocab_size=150 --mn=eurlex --e=3  --ds=EUR-Lex --data_conf="/content/DeepXML_XMLCNN/deepXML/configure/datasets/EUR-Lex.yaml" --model_conf="/content/DeepXML_XMLCNN/deepXML/configure/models/AttentionXML-EUR-Lex.yaml"
# python3 main.py --data-cnf /content/DeepXML/configure/datasets/EUR-Lex.yaml --model-cnf /content/DeepXML/configure/models/AttentionXML-EUR-Lex.yaml -t 1
# python3 main.py --data-cnf /content/DeepXML/configure/datasets/EUR-Lex.yaml --model-cnf /content/DeepXML/configure/models/AttentionXML-EUR-Lex.yaml -t 2
# python3 ensemble.py -p results/$2-$1 -t 3

#python3 /content/DeepXML_XMLCNN/deepXML/main.py --mb=32 --ss=499 --sequence_length=20 --load_data=1  --hidden_dims=3801 --vocab_size=150 --mn=eurlex --e=3  --ds=EUR-Lex --data_conf="/content/DeepXML_XMLCNN/deepXML/configure/datasets/EUR-Lex.yaml" --model_conf="/content/DeepXML_XMLCNN/deepXML/configure/models/AttentionXML-EUR-Lex.yaml" --mode="eval"


python3 /content/DeepXML_XMLCNN/deepXML/evaluation.py --results /content/DeepXML_XMLCNN/deepXML/results/AttentionXML-EUR-Lex-Tree-0-labels.npy --targets /content/drive/MyDrive/DeepXML/EUR-Lex/test_labels.npy

