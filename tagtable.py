# Import pandas

import collections
import pandas as pd
import numpy as np
# Load the xlsx file
excel_data = pd.read_excel('user-media-like.xlsx', sheet_name= 'result1')
data_tags_df = pd.read_csv('datafiletags.csv', low_memory=False)
##train_data_val = train_data.assign(rating= [1]*len(train_data))
data_tags_df_val = data_tags_df.assign(value= [1]*len(data_tags_df))
item_id_vec= data_tags_df['item_id'].unique()
user_id_vec = data_tags_df['user_id'].unique()
print (len(data_tags_df))
joined_tag_list = []
for i in range (17):
	joined_tag_list = joined_tag_list+data_tags_df["".join(['tag', str(i+1)])].dropna().to_list()
	
tags_frequency = dict (collections.Counter(joined_tag_list))
sorted_tags_frequency =dict(sorted(tags_frequency.items(), key=lambda item: item[1], reverse=True))
removed_keys = [key for key in sorted_tags_frequency.keys() if sorted_tags_frequency[key] > int (.8 *len(data_tags_df)) ]
[sorted_tags_frequency.pop(key) for key in list(sorted_tags_frequency.keys()) if sorted_tags_frequency[key] > int (.8 *len(data_tags_df)) ]
print (data_tags_df_val.shape)
print (data_tags_df.head(5))
item_id_vec= data_tags_df['item_id'].unique()
user_id_vec = data_tags_df['user_id'].unique()
vals = [0  for i in range(len(sorted_tags_frequency))]
data = [ vals for i in range(len(user_id_vec))]
label_cols = [tag for tag in sorted_tags_frequency.keys() ]
label_rows = [user for user in user_id_vec]
usertag_df = pd.DataFrame(data, label_rows, label_cols)
usertag_df1 = usertag_df.copy()
#print (data_tags_df[data_tags_df['user_id'=='user10']])
for i in range (17):
	tag_col= "".join(["tag",str(i+1)])
	data_tags_group = data_tags_df.groupby(['user_id',tag_col]).size().reset_index()
	for user , tag, tag_count in zip(data_tags_group['user_id'], data_tags_group[tag_col],data_tags_group[0]):
		if tag not in removed_keys:
			usertag_df.loc[user,tag]=usertag_df.loc[user,tag]+tag_count
	
vals = [0  for i in range(len(sorted_tags_frequency))]
data = [ vals for i in range(len(item_id_vec))]
label_cols = [tag for tag in sorted_tags_frequency.keys() ]
label_rows = [item for item in item_id_vec]
itemtag_df = pd.DataFrame(data, label_rows, label_cols)	

for i in range (17):
	tag_col= "".join(["tag",str(i+1)])
	data_tags_group = data_tags_df.groupby(['item_id',tag_col]).size().reset_index()
	for user , tag, tag_count in zip(data_tags_group['item_id'], data_tags_group[tag_col],data_tags_group[0]):
		if tag not in removed_keys:
			itemtag_df.loc[user,tag]=itemtag_df.loc[user,tag]+tag_count
print (usertag_df)
print (itemtag_df)	

