# Import Pandas
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import unique
from scipy.linalg import svd
from scipy.sparse.linalg import *
from sparsesvd import sparsesvd 
from scipy.sparse import csr_matrix
import scipy.sparse
import random
def svdcompute (mat, k):
	UT, Sigma, VT = sparsesvd(mat,k)
	S=np.diag(Sigma)
	U = csr_matrix(np.transpose(UT), dtype=np.float32)
	S = csr_matrix(S, dtype=np.float32)
	VT = csr_matrix(VT, dtype=np.float32)
	return U, S, VT


data = pd.read_csv('cleandatafile.csv', low_memory=False)
print (data.head(10))
size = data.shape
print (size)
item_id_vec= data['item_id'].unique()
user_id_vec = data['user_id'].unique()
print ("Number of unique users", len (user_id_vec))
print ("Number of unique items", len (item_id_vec))
  
# for every 5 liked item by each user, the latest is put in the test data. 
train_data_list = []
test_data_list = []
test_data=pd.DataFrame()
data_group = data.groupby('user_id')
vals = [0  for i in range(len(item_id_vec))]
data_ = [ vals for i in range(len(user_id_vec))]
label_cols = [item for item in item_id_vec]
label_rows = [user for user in user_id_vec]
useritemtrain_df = pd.DataFrame(data_, label_rows, label_cols)
useritemtest_df = useritemtrain_df.copy()
for block in data_group:
	block_sorted = block[1].sort_values('liked_date')
	n = len(block_sorted)
	n_test = int(n/4)
	n_train = n-n_test
	if n_test>0:
		train_data_list.append( block[1].iloc[:n_train])
		test_data_list.append( block[1].iloc[n_train:n])
		
		
	
train_df = pd.concat(train_data_list)
test_df = pd.concat(test_data_list)

#n_test_data.assign(rating= [1]*len(test_data))
#pd{"value": [1]*len(train_data)
##train_data_val = train_data.assign(rating= [1]*len(train_data))
##test_data_val = test_data.assign(rating= [1]*len(test_data))
#Creating a sparse pivot table with users in rows and items in columns
#users_items_pivot_matrix_df = train_data.pivot(index='user_id', columns='item_id',                                                          values=1).fillna(0)
#users_items_pivot_matrix_df.head(10)
##train_df = train_data_val.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
##test_df = test_data_val.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

#train_df_mat = train_df.as_matrix()
#test_df_mat = test_df.as_matrix()



train_data_group = train_df.groupby('user_id')
for block in train_data_group:
	for user, item in zip (block[1]['user_id'], block[1]['item_id']):
		useritemtrain_df.loc[user, item] = useritemtrain_df.loc[user,item]+1

test_data_group = test_df.groupby('user_id')
for block in test_data_group:
	for user, item in zip (block[1]['user_id'], block[1]['item_id']):
		useritemtest_df.loc[user, item] = useritemtest_df.loc[user,item]+1

train_mat=scipy.sparse.csc_matrix(useritemtrain_df.values)
k = 32
U_red, Sigma_red, VT_red = svdcompute(train_mat, k)
USVT = U_red*(Sigma_red*VT_red)#np.dot(U_reds, VT_red)
data_mat_est=USVT.toarray()

def check_item_topn(item_index, new_item_liked_filtered, topn):
	return int (item_index in new_item_liked_filtered[0:topn-1])

useritem_test_mat= useritemtest_df.to_numpy()	
useritem_est_mat= np.zeros(useritemtest_df.shape)
n =10
row = np.zeros(len(item_id_vec))
recall_at_topn_avg=0
user_id_test_vec = test_df['user_id'].unique()
#based on our traing and test spliting and data cleaning section both train df and test df have unique user_id
#for i,user in enumerate(user_id_vec):
n_predict =1000
topn =10
for user in user_id_test_vec:
	i = useritemtrain_df.index.get_loc(user)
	i_vector =-data_mat_est[i]
	sort_index=np.argsort(i_vector)
	item_liked_before_index = np.nonzero(useritemtrain_df.iloc[i].values)[0] #
	new_item_liked_index = [ sort_index[i] for i in range (len (sort_index)) if not (sort_index[i] in item_liked_before_index) ]
	index_list = list(new_item_liked_index[0:n_predict-1])
	useritem_test_row = useritem_test_mat[i]# 
	useritem_test_nzero_index = set(list(np.nonzero(useritem_test_row)[0]))
	index_test_row = set([i for i in (range(len(useritem_test_row )))])
	useritem_test_zero_index = index_test_row -  useritem_test_nzero_index 
	topn_hit =0
	for item_index in useritem_test_nzero_index:
		random_not_liked=random.choices(list(useritem_test_zero_index), k=100)
		random_not_liked.append(item_index)
		new_item_liked_filtered = [ind for ind in index_list if ind in random_not_liked]
		flag_int =  check_item_topn(item_index, new_item_liked_filtered, topn)
		topn_hit = topn_hit+flag_int
	recall_at_topn = topn_hit/float(len(useritem_test_nzero_index))
	recall_at_topn_avg = recall_at_topn+recall_at_topn_avg
		
		
		
print("*********************************")
print ("Recall@",topn,"is :",recall_at_topn_avg/len(user_id_test_vec))	
				

























