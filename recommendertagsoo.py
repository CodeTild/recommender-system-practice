import random
import collections
import pandas as pd
import numpy as np
from utilities.recommenders import *


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
		
##############################################################################################
grouping_col_str='user_id'
sorting_col_str='liked_date'
train_df, test_df = timeTestTrainsplit(data_tags_df_copy, grouping_col_str, sorting_col_str, 5)
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
usertag_mat = usertag_df.to_numpy()
itemtag_mat = itemtag_df.to_numpy()
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
useritemtest_mat= useritemtest_df.to_numpy()	
useritemtrain_mat = useritemtrain_df.to_numpy()
cf = ContentFiltering(usertag_mat, itemtag_mat, useritemtrain_mat)
predicted_mat = cf.recommendation_userblock() 
evaluation = Evaluation(useritemtrain_mat, useritemtest_mat, predicted_mat)
print(evaluation.recallat_user_block())
user_str = user_id_vec_train[0]
print ( "recommended items for", user_str, "are" , cf.recommendation_for_user(user_str, user_id_vec_train, item_id_vec_t,5))



