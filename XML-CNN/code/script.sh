#!/bin/bash
#SBATCH -n 30
#SBATCH -A nlp
#SBATCH -p long
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module add cuda/10.0
module add cudnn/7-cuda-10.0


#PYTHONUNBUFFERED=1 python main.py --mb=128  --ss=499 --sequence_length=5000 --load_data=1  --hidden_dims=1024 --vocab_size=100000 --mn=wiki --e=500  --ds=Wiki10-31K > out_wiki.txt

PYTHONUNBUFFERED=1 python main.py --sequence_length=5000 --load_data=1  --hidden_dims=1024 --vocab_size=100000 --mn=wiki  --ds=Wiki10-31K --lm=../saved_models/wiki10/model_best_test --tr=0 > out_wiki_test.txt
#0python main.py --ss=500 --sequence_length=5000 --load_data=1  --hidden_dims=1024 --vocab_size=100000 --mn=wiki --e=1000  --ds=Wiki10-31K > out_wiki.txt
#python main.py --sequence_length=5000 --lm=../saved_models/wiki/model_best_test --hidden_dims=1024 --vocab_size=100000 --mn=wiki --e=100  --ds=Wiki10-31K --tr=0

#python main.py --mb=128 --ss=499 --sequence_length=10000 --load_data=1  --hidden_dims=1024 --vocab_size=150000 --mn=eurlex --e=500  --ds=Eurlex  > out_eurlex.txt 

#python main.py --ss=250 --sequence_length=5000 --load_data=1  --hidden_dims=1024 --vocab_size=100000 --mn=eurlex --e=500  --ds=EUR-Lex --lm=../saved_models/eurlex/model_best_test --tr=0

#PYTHONUNBUFFERED=1 python /home/nilabja.bhattacharya/MTP2020-RankingXML/main.py > output_rcv.txt
