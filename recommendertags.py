
import random
from sklearn.metrics.pairwise import cosine_similarity
import collections
import pandas as pd
import numpy as np
data_tags_df = pd.read_csv('datafiletags.csv', low_memory=False)

#data_tags_df_val = data_tags_df.assign(value= [1]*len(data_tags_df))

joined_tag_list = []
for i in range (17):
	joined_tag_list = joined_tag_list+data_tags_df["".join(['tag', str(i+1)])].dropna().to_list()
	
tags_frequency = dict (collections.Counter(joined_tag_list))
sorted_tags_frequency =dict(sorted(tags_frequency.items(), key=lambda item: item[1], reverse=True))
# Tags with high rate of occurence do not provide useful information. 
removed_keys = [key for key in sorted_tags_frequency.keys() if sorted_tags_frequency[key] > int (.8 *len(data_tags_df)) ]
[sorted_tags_frequency.pop(key) for key in list(sorted_tags_frequency.keys()) if sorted_tags_frequency[key] > int (.8 *len(data_tags_df)) ]
item_id_vec= data_tags_df['item_id'].unique()
user_id_vec = data_tags_df['user_id'].unique()
vals = [0  for i in range(len(sorted_tags_frequency))]
data = [ vals for i in range(len(item_id_vec))]
label_cols = [tag for tag in sorted_tags_frequency.keys() ]
label_rows = [item for item in item_id_vec]
itemtag_df = pd.DataFrame(data, label_rows, label_cols)	
###############################################################################################################
data_tags_group = data_tags_df.groupby(['item_id']).size().reset_index()
for i in range (17):
	tag_col= "".join(["tag",str(i+1)])
	data_tags_group = data_tags_df.groupby(['item_id',tag_col]).size().reset_index()
	#print (data_tags_group)
	for item , tag, tag_count in zip(data_tags_group['item_id'], data_tags_group[tag_col],data_tags_group[0]):
		if tag not in removed_keys:
			itemtag_df.loc[item,tag]=itemtag_df.loc[item,tag]+tag_count
#################################################################################################################
#remove items and  users without tags 
#Some items and users may not have any tags assigned becuse some tages are removed
sum_row_items= itemtag_df.sum(axis='columns')
items=sum_row_items[sum_row_items==0]
items_index = list(items.index)

data_tags_df_copy = data_tags_df.copy()
if (len (items)>0 ):
	itemtag_df.drop(items_index,axis=0,inplace=True)	
	[data_tags_df_copy.drop(data_tags_df_copy[data_tags_df_copy.item_id == item].index, inplace=True) for item in items_index]
		
#################################################################################################################
#################################################################################################################
train_data_list =[]
test_data_list= []
train_data_dict={}
test_data_dict={}
data_group = data_tags_df_copy.groupby('user_id')
for block in data_group:
	block_sorted = block[1].sort_values('liked_date')	
	n = len(block_sorted)
	n_test = int(n/4)
	n_train = n-n_test
	if int(n/5)>0: #just keep users that have 5 like items
		train_data_list.append( block[1].iloc[:n_train])
		test_data_list.append( block[1].iloc[n_train:n])
		train_data_dict[block[0]]= list(block[1].iloc[:n_train]['item_id']) #?
		test_data_dict[block[0]]= list (block[1].iloc[n_train:n]['item_id']) #?
train_df = pd.concat(train_data_list)
test_df = pd.concat(test_data_list)
###############################################################################################################################
user_id_vec_train = train_df['user_id'].unique()
user_id_vec_test = test_df['user_id'].unique()

vals = [0  for i in range(len(sorted_tags_frequency))]
data = [ vals for i in range(len(user_id_vec_train))]
label_cols = [tag for tag in sorted_tags_frequency.keys() ]
label_rows = [user for user in user_id_vec_train]
usertag_df = pd.DataFrame(data, label_rows, label_cols)
for user in user_id_vec_train:
	for i in range (17):
		tag_col= "".join(["tag",str(i+1)])		
		data_tags_group = train_df[train_df['user_id']==user].groupby(tag_col).size().reset_index()
		for tag, tag_count  in zip (data_tags_group[tag_col], data_tags_group[0]):
			if tag not in removed_keys:
				usertag_df.loc[user,tag]=usertag_df.loc[user,tag]+tag_count
#################################################################################################
#Can we find a user without any labe?
sum_row_users = usertag_df.sum(axis='columns')
users=sum_row_users[sum_row_users==0]
users_index = list(users.index)
usertag_mat = usertag_df.to_numpy()
itemtag_mat = itemtag_df.to_numpy()
##############################################################
##Noramlization method 1	
#)

size = itemtag_mat.shape
sum_rows= np.sum(itemtag_mat, axis = 1, keepdims = True)
sum_mat = np.hstack([sum_rows]*size[1])
tf_mat= np.divide(itemtag_mat,sum_mat)
n_nzeros = np.count_nonzero(itemtag_mat, axis=0)
sum_mat = np.vstack([n_nzeros]*size[0])
idf_mat = np.log10(np.divide( size[0]*np.ones(itemtag_mat.shape), 1+sum_mat))
tf_idf_mat_item= np.multiply(tf_mat,idf_mat)

##############################################################
size = usertag_mat.shape
sum_rows= np.sum(usertag_mat, axis = 1, keepdims = True)
sum_mat = np.hstack([sum_rows]*size[1])
tf_mat= np.divide(usertag_mat,sum_mat)
n_nzeros = np.count_nonzero(usertag_mat, axis=0)
sum_mat = np.vstack([n_nzeros]*size[0])
idf_mat = np.log10(np.divide( size[0]*np.ones(usertag_mat.shape), 1+sum_mat))
tf_idf_mat_user= np.multiply(tf_mat,idf_mat)
similarity_mat = cosine_similarity(tf_idf_mat_user,tf_idf_mat_item)
sorted_index_mat = np.argsort(-similarity_mat)
def check_item_topn(item_index, new_item_liked_filtered, topn):
	return int (item_index in new_item_liked_filtered[0:topn-1])

##################################################################
#producing user-item-data frame train and test 
user_id_vec_train = train_df['user_id'].unique()
user_id_vec_test = test_df['user_id'].unique()
item_id_vec_t = data_tags_df_copy['item_id'].unique() # include items that migh not be in train and test 
#item_id_vec_t includes both train and test and more 
#################################################################
vals = [0  for i in range(len(item_id_vec_t))]
data = [ vals for i in range(len(user_id_vec_train))]
label_cols = [item  for item  in item_id_vec_t]
label_rows = [user for user in user_id_vec_train]
useritemtrain_df = pd.DataFrame(data, label_rows, label_cols)
#pd.pivot pivot_table
#
for user, item  in zip(train_df['user_id'], train_df['item_id']):	
	useritemtrain_df.loc[user,item]=useritemtrain_df.loc[user,item]+1
vals = [0  for i in range(len(item_id_vec_t))]
data = [ vals for i in range(len(user_id_vec_train))]
label_cols = [item  for item  in item_id_vec_t]
label_rows = [user for user in user_id_vec_train]
useritemtest_df = pd.DataFrame(data, label_rows, label_cols)


for user, item  in zip(test_df['user_id'], test_df['item_id']):	
	useritemtest_df.loc[user,item]=useritemtest_df.loc[user,item]+1
####################################################################
useritem_test_mat= useritemtest_df.to_numpy()	
useritemtrain_mat = useritemtrain_df.to_numpy()
useritem_pred_mat= np.zeros(useritemtest_df.shape)
n =10
row = np.zeros(len(item_id_vec))
recall_at_topn_avg=0
user_id_test_vec = test_df['user_id'].unique()
index_total_set= set([i for i in range (len(item_id_vec_t))])
#based on our traing and test spliting and data cleaning section both train df and test df have same unique user_id

n_predict =1000
topn =10

for user in user_id_test_vec:
	i = usertag_df.index.get_loc(user)
	sort_index =sorted_index_mat[i]
	item_liked_before_index = np.nonzero(useritemtrain_df.iloc[i].values)[0] #
	#new_item_liked_index = [ sort_index[i] for i in range (len (sort_index)) if not (sort_index[i] in item_liked_before_index) ]
	index_list = list(sort_index[0:n_predict-1])
	useritem_test_row = useritem_test_mat[i]# 
	useritem_test_nzero_index = set(list(np.nonzero(useritem_test_row)[0]))
	useritem_test_zero_index = index_total_set -  useritem_test_nzero_index - set(item_liked_before_index) 
	topn_hit =0
	for item_index in useritem_test_nzero_index:
		random_not_liked=random.sample(list(useritem_test_zero_index), k=100)
		random_not_liked.append(item_index)
		new_item_liked_filtered = [ind for ind in sort_index if ind in random_not_liked]
		flag_int =  check_item_topn(item_index, new_item_liked_filtered, topn)
		topn_hit = topn_hit+flag_int
	recall_at_topn = topn_hit/float(len(useritem_test_nzero_index))
	recall_at_topn_avg = recall_at_topn+recall_at_topn_avg

print("*********************************")
print ("Recall@",topn,"is :",recall_at_topn_avg/len(user_id_test_vec))	




