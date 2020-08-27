import torch
from data import IntentDset
from model import ProtNet 
from torch import nn, optim 
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='lena', type=str)
parser.add_argument('--dev_data', default='moli', type=str)
parser.add_argument('--seq_len', default='10', type=int)
parser.add_argument('--train_batch_size', default='10', type=int)
parser.add_argument('--dev_batch_size', default='10', type=int)
opt = parser.parse_args()

# https://github.com/cyvius96/prototypical-network-pytorch/blob/master/utils.py
def euclidean_metric(a, b):
	n = a.shape[0]
	m = b.shape[0]
	a = a.unsqueeze(1).expand(n, m, -1)
	b = b.unsqueeze(0).expand(n, m, -1)
	logits = -((a - b)**2).sum(dim=2)
	return logits

N_c_tr = 57
N_c_te = 297 
if opt.train_data =='lena':
    N_c_tr = 57
    N_c_te = 297 
elif opt.train_data =='moli':
    N_c_tr = 297
    N_c_te = 57 

N_i_tr = 1
N_i_te = 1

N_q_tr = 1
N_q_te = 1

idset = IntentDset(dataset = 'data', split = opt.train_data, Nc = N_c_tr, Ni = N_i_tr, n_query = N_q_tr, seq_len = opt.seq_len)
val_dset = IntentDset(dataset = 'data', split = opt.dev_data, Nc = N_c_te, Ni = N_i_te, n_query = N_q_te, seq_len = opt.seq_len)

pn = ProtNet().cuda()
pn = nn.DataParallel(pn)
pn = pn.cuda()

param_optimizer = list(pn.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
					lr=2e-5,
					warmup=0.1,
					t_total=10000)

criterion = nn.CrossEntropyLoss()

step = 0

best_accuracy = 0
best_step = 0
while True:
	pn.train()
	step += 1
    
	train_batch_size = opt.train_batch_size
	batch_num = math.ceil(len(idset.sents) / train_batch_size)
	for i in range(batch_num):
		begin = i * train_batch_size
		end = min((i+1) * train_batch_size, len(idset.sents))
		qry_batch = idset.sents[begin: end]
		query_sens = []
		query_label = []
		for j in range(begin, end):
			if idset.labels[j] not in idset.unqiue_labels:
				qry_batch.remove(qry_batch[j % train_batch_size])
				continue
			idset.label_bins[idset.label2id[idset.labels[j]]].remove(idset.sents[j])
			query_label.append(idset.label2id[idset.labels[j]])
			query_sens.append(idset.sents[j])
		target_input_ids = torch.tensor([f.input_ids for f in qry_batch], dtype=torch.long)
		target_input_mask = torch.tensor([f.input_mask for f in qry_batch], dtype=torch.long)
		qry_set = {}
		qry_set['input_ids'] = target_input_ids
		qry_set['input_mask'] = target_input_mask
                
		batch, n_labels = idset.next_batch_test_full()
		sup_set = batch['sup_set_x']
		for j in range(len(query_sens)):
			idset.label_bins[query_label[j]].append(query_sens[j])

		sup = pn(sup_set['input_ids'].cuda(),sup_set['input_mask'].cuda())
		qry = pn(qry_set['input_ids'].cuda(),qry_set['input_mask'].cuda())
		sup = sup.view(n_labels, N_i_tr,-1).mean(1)
		logits = euclidean_metric(qry, sup)
		label = torch.tensor(query_label).type(torch.LongTensor).cuda()
		loss = criterion(logits, label)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		print('Iteration :',step,"Loss :",float(loss.item()))
    
	if step%1 == 0:
		pn.eval()
		pn.cuda()
		total = 0
		correct = 0
		for i in range(1):
			dev_batch_size = opt.dev_batch_size
			batch_num = math.ceil(len(val_dset.sents) / dev_batch_size)
			for i in range(batch_num):
				begin = i * dev_batch_size
				end = min((i+1) * dev_batch_size, len(val_dset.sents))
				qry_batch = val_dset.sents[begin: end]
				query_sens = []
				query_label = []
				for j in range(begin, end):
					if val_dset.labels[j] not in val_dset.unqiue_labels:
						qry_batch.remove(qry_batch[j % dev_batch_size])
						continue
					val_dset.label_bins[val_dset.label2id[val_dset.labels[j]]].remove(val_dset.sents[j])
					query_label.append(val_dset.label2id[val_dset.labels[j]])
					query_sens.append(val_dset.sents[j])
                
				target_input_ids = torch.tensor([f.input_ids for f in qry_batch], dtype=torch.long)
				target_input_mask = torch.tensor([f.input_mask for f in qry_batch], dtype=torch.long)
				qry_set = {}
				qry_set['input_ids'] = target_input_ids
				qry_set['input_mask'] = target_input_mask
                
				batch, n_labels = val_dset.next_batch_test_full()
				sup_set = batch['sup_set_x']
				for j in range(len(query_sens)):
					val_dset.label_bins[query_label[j]].append(query_sens[j])

				sup = pn(sup_set['input_ids'].cuda(),sup_set['input_mask'].cuda())
				qry = pn(qry_set['input_ids'].cuda(),qry_set['input_mask'].cuda())
				sup = sup.view(n_labels, N_i_te,-1).mean(1)
				logits = euclidean_metric(qry, sup).max(1)[1].cpu()
				label = torch.tensor(query_label).type(torch.LongTensor)
				correct += float(torch.sum(logits==label).item())
				total += label.shape[0]
		print(correct,'/',total)
		print('Accuracy :',correct/total)
		if (correct/total) > best_accuracy:
			best_accuracy = correct/total
			best_step = step
		print('Best accuracy :', best_accuracy)
		print('Best step :', best_step)
		pn.cuda()
	if step%100000 == 0:
		break
