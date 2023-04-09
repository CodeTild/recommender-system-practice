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

def timeTestTrainsplit( df, grouping_col_name, sorting_col_name, n_t): # 1 test and n-1 train
	train_data_list =[]
	test_data_list= []
	#train_data_dict={}
	#test_data_dict={}
	data_group = df.groupby(grouping_col_name)
	for block in data_group:
		block_sorted = block[1].sort_values(sorting_col_name)	
		n = len(block_sorted)
		n_test = int(n/4)
		n_train = n-n_test
		if int(n/n_t)>0: #just keep users that have 5 like items for svd recommender this condotion is redandunt
			train_data_list.append( block[1].iloc[:n_train])
			test_data_list.append( block[1].iloc[n_train:n])
			#train_data_dict[block[0]]= list(block[1].iloc[:n_train]['item_id']) #?
			#test_data_dict[block[0]]= list (block[1].iloc[n_train:n]['item_id']) #?
	train_df = pd.concat(train_data_list)
	test_df = pd.concat(test_data_list)
	return train_df, test_df
	

def data_frame_manuplation(item_id_vec, user_id_vec , train_df, test_df ):
	vals = [0  for i in range(len(item_id_vec))]
	data_ = [ vals for i in range(len(user_id_vec))]
	label_cols = [item for item in item_id_vec]
	label_rows = [user for user in user_id_vec]
	useritemtrain_df = pd.DataFrame(data_, label_rows, label_cols)
	useritemtest_df = useritemtrain_df.copy()
	train_data_group = train_df.groupby('user_id')
	for block in train_data_group:
		for user, item in zip (block[1]['user_id'], block[1]['item_id']):
			useritemtrain_df.loc[user, item] = useritemtrain_df.loc[user,item]+1

	test_data_group = test_df.groupby('user_id')
	for block in test_data_group:
		for user, item in zip (block[1]['user_id'], block[1]['item_id']):
			useritemtest_df.loc[user, item] = useritemtest_df.loc[user,item]+1
	return useritemtrain_df, useritemtest_df
	
def tf_idf_mat(xxxtag_mat):
	size = xxxtag_mat.shape
	sum_rows= np.sum(xxxtag_mat, axis = 1, keepdims = True)
	sum_mat = np.hstack([sum_rows]*size[1])
	tf_mat= np.divide(xxxtag_mat,sum_mat)
	n_nzeros = np.count_nonzero(xxxtag_mat, axis=0)
	sum_mat = np.vstack([n_nzeros]*size[0])
	idf_mat = np.log10(np.divide( size[0]*np.ones(xxxtag_mat.shape), 1+sum_mat))
	tf_idf_mat= np.multiply(tf_mat,idf_mat)
	return tf_idf_mat

class Evaluation:
	def __init__(self, useritemtrain_mat, useritemtest_mat, predicted_mat):
		self.useritemtest_mat = useritemtest_mat
		self.useritemtrain_mat = useritemtrain_mat
		self.predicted_mat = predicted_mat
	##__init
	def check_item_topn(self, item_index, new_item_liked_filtered, topn):
		return int (item_index in new_item_liked_filtered[0:topn-1])
	def recallat_user(self, sort_index, item_liked_before_index, useritem_test_row,  n_predict, topn , k ):
		total_item_index = set ([i for i in range(len(sort_index))]) 
		index_list = list(sort_index[0:n_predict-1])
		useritem_test_nzero_index = set(np.nonzero(useritem_test_row)[0])
		useritem_test_zero_index = total_item_index -  useritem_test_nzero_index -  set(item_liked_before_index)
		topn_hit =0			
		for item_index in useritem_test_nzero_index:
			random_not_liked=random.sample(list(useritem_test_zero_index), k)
			random_not_liked.append(item_index)
			new_item_liked_filtered = [ind for ind in index_list if ind in random_not_liked]
			flag_int =  self.check_item_topn(item_index, new_item_liked_filtered, topn)
			topn_hit = topn_hit+flag_int
			recall_at_topn = topn_hit/float(len(useritem_test_nzero_index))
		return recall_at_topn
	def recallat_user_block (self, n_predict =1000, topn =10, k=100 ):
		recall_at_topn_total=0
		for i in range(self.useritemtest_mat.shape[0]):
			sort_index = self.predicted_mat[i]
			item_liked_before_index = np.nonzero(self.useritemtrain_mat[i])[0]
			useritem_test_row = self.useritemtest_mat[i]
			recall_at_topn_total = recall_at_topn_total+self.recallat_user(sort_index, item_liked_before_index, useritem_test_row, n_predict, topn , k )
		return recall_at_topn_total/self.useritemtest_mat.shape[0]     

class SVDrecommender:

	def __init__(self, user_item_mat,k_svd ):
        	self.user_item_mat = user_item_mat
        	self.k_svd = k_svd
        	        	
	def svd_recommender_userblock (self):
		UT, Sigma, VT = sparsesvd(self.user_item_mat, self.k_svd)
		S=np.diag(Sigma)
		U = csr_matrix(np.transpose(UT), dtype=np.float32)
		S = csr_matrix(S, dtype=np.float32)
		VT = csr_matrix(VT, dtype=np.float32)
		USVT = U*(S*VT)
		user_item_est_mat=USVT.toarray()
		self.predicted_mat =np.argsort(-user_item_est_mat)
		return self.predicted_mat#, U, S, VT
	def svd_recommender_user (self, user_id,user_id_vec,item_id_vec, n_predict):
		i=list(user_id_vec).index(user_id)
		index_list= self.predicted_mat[i,0:n_predict-1]
		recommended_items = [item_id_vec[ind] for ind in index_list]
		return recommended_items
			
		
		
