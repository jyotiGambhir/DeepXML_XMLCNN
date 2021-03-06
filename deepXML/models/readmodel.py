###For AttentionXML Eur-Lex
'''
emb.emb.weight  :  (166402, 300)
lstm.init_state  :  (4, 1, 256)
lstm.lstm.weight_ih_l0  :  (1024, 300)
lstm.lstm.weight_hh_l0  :  (1024, 256)
lstm.lstm.bias_ih_l0  :  (1024,)
lstm.lstm.bias_hh_l0  :  (1024,)
lstm.lstm.weight_ih_l0_reverse  :  (1024, 300)
lstm.lstm.weight_hh_l0_reverse  :  (1024, 256)
lstm.lstm.bias_ih_l0_reverse  :  (1024,)
lstm.lstm.bias_hh_l0_reverse  :  (1024,)
attention.attention.weight  :  (3801, 512)
linear.linear.0.weight  :  (256, 512)
linear.linear.0.bias  :  (256,)
linear.output.weight  :  (1, 256)
linear.output.bias  :  (1,)
'''
##### For FastAttentionXML Model Amazon 670K
'''
Network.emb.emb.weight  :  (500000, 300)
Network.lstm.init_state  :  (4, 1, 512)
Network.lstm.lstm.weight_ih_l0  :  (2048, 300)
Network.lstm.lstm.weight_hh_l0  :  (2048, 512)
Network.lstm.lstm.bias_ih_l0  :  (2048,)
Network.lstm.lstm.bias_hh_l0  :  (2048,)
Network.lstm.lstm.weight_ih_l0_reverse  :  (2048, 300)
Network.lstm.lstm.weight_hh_l0_reverse  :  (2048, 512)
Network.lstm.lstm.bias_ih_l0_reverse  :  (2048,)
Network.lstm.lstm.bias_hh_l0_reverse  :  (2048,)
Network.attention.attention.weight  :  (16385, 1024)
Network.linear.linear.0.weight  :  (512, 1024)
Network.linear.linear.0.bias  :  (512,)
Network.linear.linear.1.weight  :  (256, 512)
Network.linear.linear.1.bias  :  (256,)
Network.linear.output.weight  :  (1, 256)
Network.linear.output.bias  :  (1,)
''' 

import numpy as np
import torch

model_name = "AttentionXML-Wiki10-31K-Tree-0"

t = torch.load(model_name,map_location='cpu')

for key, val in t.items():
    print(key," : ",val.cpu().detach().numpy().shape)


attn_weight = t["attention.attention.weight"]
aw = attn_weight.cpu().detach().numpy()
np.save("Wiki10_attn_weight",aw)
print(aw.shape)