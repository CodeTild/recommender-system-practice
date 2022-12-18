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
def svdcompute (mat, k):
	UT, Sigma, VT = sparsesvd(mat,k)
	S=np.diag(Sigma)
	U = csr_matrix(np.transpose(UT), dtype=np.float32)
	S = csr_matrix(S, dtype=np.float32)
	VT = csr_matrix(VT, dtype=np.float32)
	return U, S, VT

# Load Movies Metadata
data = pd.read_csv('user_like_new.csv', low_memory=False)
data.head()
size = data.shape
print(data)  
print (data.shape)
print(data.head())
media_id_vec= data['media_id'].unique()
user_id_vec = data['user_id'].unique()

train_data_list = []
test_data_list = []
test_data=pd.DataFrame()
for user in user_id_vec:
	temp = data[data['user_id'] == user]
	n = len(temp)
	n_test = int(n/4)
	n_train = n-n_test
	train_data_list.append( temp.iloc[:n_train])
	if n_test>0:
		test_data_list.append( temp.iloc[n_train:n])
train_data = pd.concat(train_data_list)
test_data = pd.concat(test_data_list)
media_id_un= train_data['media_id'].unique()
user_id_un = train_data['user_id'].unique()

vals = [0  for i in range(len(user_id_vec))]
data = [ vals for i in range(len(media_id_vec))]
label_cols = [user for user in user_id_vec]
label_rows = [med_item for med_item in media_id_vec]
new_df = pd.DataFrame(data, label_rows, label_cols)
for i, user in enumerate (user_id_vec):#(new_df['user_id'].values):
	seq=train_data[train_data['user_id'] == user]['media_id']
	for item in seq:
		new_df.loc[item,user] = new_df.loc[item,user]+1
#producing user-item  user based colabrative filtering 
vals = [0  for i in range(len(media_id_vec))]
data = [ vals for i in range(len(user_id_vec))]
label_cols = [media for media in media_id_vec]
label_rows = [user for user in user_id_vec]
usermedia_df = pd.DataFrame(data, label_rows, label_cols)
for i, user in enumerate (user_id_vec):#(new_df['user_id'].values):
	seq=train_data[train_data['user_id'] == user]['media_id']
	for item in seq:
		new_df.loc[item,user] = new_df.loc[item,user]+1
for user in user_id_vec:#(new_df['user_id'].values):
	seq=train_data[train_data['user_id'] == user]['media_id']
	for item in seq:
		usermedia_df.loc[user,item] = usermedia_df.loc[user,item]+1

mat=scipy.sparse.csc_matrix(usermedia_df.values)
U_red, Sigma_red, VT_red = svdcompute(mat, 200)

USVT = U_red*(Sigma_red*VT_red)#np.dot(U_reds, VT_red)

th=5

usermedia_df_es= USVT >=th*np.ones(USVT.shape)
error_sum =0
size=usermedia_df_es.shape
error_sum = 0;
len(test_data)

for user in test_data['user_id'].values:
	seq=test_data[test_data['user_id'] == user]['media_id']
	for media in seq:
		med_index= np.where(media_id_vec==media)[0][0]
		user_index = np.where(user_id_vec==user)[0][0]
		if (usermedia_df_es[user_index,med_index]!=usermedia_df.loc[user,item]):
			error_sum +=1
		
print (error_sum/len(test_data))
error_sum2=0
for user in user_id_vec:
	for media in media_id_vec:
		if (usermedia_df_es[user_index,med_index]!=usermedia_df.loc[user,item]):
			error_sum2 +=1	

print (error_sum2/(len(test_data)+len (train_data)))


