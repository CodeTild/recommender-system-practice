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
	#U, s, Vt = sparsesvd(urm, K
	#Sigma_red = Sigma[0:k]
	#U_red = U[:, 0:k]
	#VT_red = VT[0:k,:]
	#dim = (len(s), len(s))
	S=np.diag(Sigma)
	#S = np.zeros(dim, dtype=np.float32)
	#for i in range(0, len(s)):
	#	S[i,i] = mt.sqrt(s[i])

	U = csr_matrix(np.transpose(UT), dtype=np.float32)
	S = csr_matrix(S, dtype=np.float32)
	VT = csr_matrix(VT, dtype=np.float32)
	return U, S, VT

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
print (type(media_id_vec))
###########################################################################################################################################
#Data Analysis:
#Does a user view an item more than once.
#for u_item in user_id_vec:
#	mforu_valus=data[data['user_id']== u_item]['media_id'].unique()
#	mforu_greq=data[data['user_id']== u_item]['media_id'].unique()
#
################################################################################################################################
#Sort data set in terms of time stamps




#producing test data test_ratio = 0.2 #fraction of data to be used as test set.for u in users:
#test_ratio =0.2
train_data_list = []
test_data_list = []
test_data=pd.DataFrame()
for user in user_id_vec:
	#print("******")	
	#print("item",user)
	temp = data[data['user_id'] == user]
	#print("temp", temp)
	n = len(temp)
	#print(n)
	n_test = int(n/4)
	#print (n_test)
	n_train = n-n_test
	#print(n_train)
	train_data_list.append( temp.iloc[:n_train])
	#print ("train_data", train_data)
	if n_test>0:
		test_data_list.append( temp.iloc[n_train:n])
	#print ("test_data", train_data_list)
train_data = pd.concat(train_data_list)
test_data = pd.concat(test_data_list)
#print(train_data.shape)
#print(test_data.shape)  		


media_id_un= train_data['media_id'].unique()

user_id_un = train_data['user_id'].unique()
#print (len (media_id_un))
#print (len (media_id_vec))
#print (len (user_id_un))
#print (len (user_id_vec))
vals = [0  for i in range(len(user_id_vec))]
data = [ vals for i in range(len(media_id_vec))]
label_cols = [user for user in user_id_vec]
#print (len(label_cols))
#print(len(user_id_un))
#print(len(vals))
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
#print (len(label_cols))
#print(len(user_id_un))
#print(len(vals))
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
#df = usermedia_df.sparse.to_coo()		
#mat=usermedia_df.scipy.sparse.csc_matrix
aa=usermedia_df.to_numpy
#print(aa)
mat=scipy.sparse.csc_matrix(usermedia_df.values)
#print(type(mat))
#print(mat)
#mat=csr_matrix(usermedia_df.astype(usermedia_df.SparseDtype("float64",0)).sparse.to_coo())		
U_red, Sigma_red, VT_red = svdcompute(mat, 200)
#print(U_red.shape)
#print(Sigma_red.shape)
#print(VT_red.shape)
#print(usermedia_df.shape)
#print(sigma_red.shape)
#print(VT_red.shape)
#print(right.shape)
#s=np.diag(Sigma_red)
#U_reds=np.dot(U_red,s)
USVT = U_red*(Sigma_red*VT_red)#np.dot(U_reds, VT_red)
#print(U_reds.shape)
#right=s*VT_red
#print(sigma_red.shape)
#print(VT_red.shape)
#print(right.shape)
#print(U_red.shape)
#estimatedum_df=U_red*right
th=.5
#(df[df[‘Name’]==’Donna’].index.valuesUSVT
print (USVT)
usermedia_df_es= USVT >=th*np.ones(USVT.shape)
#print (usermedia_df_es)
#print (usermedia_df_es)
error_sum =0
#print(len (media_id_vec))
#print(len (user_id_vec))
#print(usermedia_df_es.shape)
#print(usermedia_df.shape)
size=usermedia_df_es.shape
#print(type(usermedia_df_es))
#mat=usermedia_df_es.to_numpy
#print(size[0])
print(size[1])
#for i in range(size[0]):
#	for j in range(size[1]):
#		if (usermedia_df_es[i,j]):
#			print(i,j)
#			print (usermedia_df_es[i,j])
error_sum = 0;
len(test_data)

for user in test_data['user_id'].values:
	seq=test_data[test_data['user_id'] == user]['media_id']
	for media in seq:
		med_index= np.where(media_id_vec==media)[0][0]
		user_index = np.where(user_id_vec==user)[0][0]
		#print(med_index)
		#print(user_index)
		if (usermedia_df_es[user_index,med_index]!=usermedia_df.loc[user,item]):
			error_sum +=1
		
print (error_sum/len(test_data))
error_sum2=0
for user in user_id_vec:
	for media in media_id_vec:
		#med_index= np.where(media_id_vec==media)[0][0]
		#user_index = np.where(user_id_vec==user)[0][0]
		#print(med_index)
		#print(user_index)
		if (usermedia_df_es[user_index,med_index]!=usermedia_df.loc[user,item]):
			error_sum2 +=1	

print (error_sum2/(len(test_data)+len (train_data)))

#print (Sigma_red)
#print (Sigma_red.shape)			
	
#print(new_df)	 

#print(new_df.to_string())			
