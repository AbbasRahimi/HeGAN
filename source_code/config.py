batch_size = 16
lambda_gen = 1e-3
lambda_dis = 1e-3
n_sample = 16
lr_gen = 0.01#1e-3
lr_dis = 0.001#1e-4
n_epoch = 10
saves_step = 5
sig = 1.0

label_smooth = 0.0

d_epoch = 15
g_epoch = 5

n_emb = 16

# dataset = 'yelp'
# dataset = 'dblp'
dataset = 'family'

#graph_filename = '../data/' + dataset + '/' + dataset + '_triple.dat'
graph_filename = '/data/' + dataset + '_triple.dat'


pretrain_node_emb_filename_d = '/pre_train/node_clustering/' + dataset + '_pre_train.emb'
pretrain_node_emb_filename_g = '/pre_train/node_clustering/' + dataset + '_pre_train.emb'
#pretrain_rel_emb_filename_d = '../data/' + dataset + '/rel_embeddings.txt'
#depretrain_rel_emb_filename_g = '../data/' + dataset + '/rel_embeddings.txt'

emb_filenames = ['/results/' + dataset + '_gen.emb',
                 '/results/' + dataset + '_dis.emb']

model_log = '/log/'
