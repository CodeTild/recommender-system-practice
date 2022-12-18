# Import Pandas
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import unique

# Load Movies Metadata
data = pd.read_csv('user_like_new.csv', low_memory=False)

# Print the first three rows
data.head()
#print(data.to_string())
size = data.shape
print(data)  
print (data.shape)
print(data.head())
media_id_vec= data['media_id'].unique()
#print(len(media_id_vec))

user_id_vec = data['user_id'].unique()
###########################################################################################################################################
#Data Analysis:
#Does a user view an item more than once.
for u_item in user_id_vec:
	mforu_valus=data[data['user_id']== u_item]['media_id'].unique()
	mforu_greq=data[data['user_id']== u_item]['media_id'].unique()
	#print(type(mforu_list))
	#unique_vals, unique_ind, unique_freq =mforu_list.unique(return_index=True, return_counts=True)

#print(len(user_id_vec))

#creare test dataset in terms of time 

#df['sepal width (cm)'] < 3
n = size[0]
ratio =1-0.2
n_train = int(size[0]*ratio)
n_test = n-n_train

train_data = data.iloc[:n_train-1]
test_data = data.iloc[n_train:n]
#print(train_subset.to_string())
#print(test_subset.to_string())

media_id_un= train_data['media_id'].unique()
#print(len(media_id_vec))

user_id_un = train_data['user_id'].unique()
#print(len(user_id_vec))

#new_df=pd.DataFrame(user_id_un)
#new_df.columns=['user_id']
#new_df.rows= ['media_id']
#new_df.columns=[_id']

#keys = [med_item for med_item in media_id_un]
#vals = [0  for i in range(len(user_id_un)) ]
#new_data={
#	'user_id' : [user for user in user_id_un]}

#for item in media_id_un:
#	new_data[item]= vals
#new_df=pd.DataFrame(new_data) 
vals = [0  for i in range(len(user_id_un))]
data = [ vals for i in range(len(media_id_un))]
label_cols = [user for user in user_id_un]
print (len(label_cols))
print(len(user_id_un))
print(len(vals))
label_rows = [med_item for med_item in media_id_un]




#print(len(label_cols))
#print(len(label_rows))
#print(len(media_id_un))
#print(len(user_id_vec))
new_df = pd.DataFrame(data, label_rows, label_cols)
#number_neighbors = 3 
#knn = NearestNeighbors(metric='cosine', algorithm='brute')
#knn.fit(new_df.values)
#distances, indices = knn.kneighbors(new_df.values, n_neighbors=number_neighbors)
#print(new_df)
#for media in media_id_un:
#	for user in user_id_un:
#		new_df.loc[user,media]=0
		
		
	#print(new_df[user])	
	
	#item for item 
	#print(item)
	
#print (new_df)
#pd.concat([item for item in media_id_un], axis =1)
#print(new_df)	
#print(new_df)
for i, user in enumerate (user_id_un):#(new_df['user_id'].values):
	seq=train_data[train_data['user_id'] == user]['media_id']
	for item in seq:
		new_df.loc[item,user] = new_df.loc[item,user]+1
		#if (new_df.loc[user,item]>1):
		#	print(new_df.loc[user,item])
		#new_df.set_value(user, item, new_df[user][item]+1)
	#for j, media_item in enumerate(media_id_un):
		#train_data[df['user_id'] == 2]['B']
#print("**********************")		
#for user in user_id_un:
#	for item in media_id_un:
#		if (new_df.loc[user,item]>1):
#			print (new_df.loc[user, item])
#			seq=train_data[train_data['user_id'] == user]['media_id']
#			count = 0;
#			for s in seq:
#				if ( s == item ):
#					count = count +1
#			print (count)			
#			print ("*********")
	
			
	
print(new_df)	 

#print(new_df.to_string())			
