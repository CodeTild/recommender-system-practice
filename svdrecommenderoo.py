import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse
from utilities.recommenders import *



data = pd.read_csv('cleandatafile.csv', low_memory=False)
print (data.head(10))
size = data.shape
print (size)
item_id_vec= data['item_id'].unique()
user_id_vec = data['user_id'].unique()
print ("Number of unique users", len (user_id_vec))
print ("Number of unique items", len (item_id_vec))
  

grouping_col_str='user_id'
sorting_col_str='liked_date'

train_df, test_df = timeTestTrainsplit(data, grouping_col_str, sorting_col_str, 5)
print (train_df)
print (test_df)

useritemtrain_df, useritemtest_df=data_frame_manuplation(item_id_vec, user_id_vec , train_df, test_df )
ueritemtrain_mat=scipy.sparse.csc_matrix(useritemtrain_df.values)
model = SVDrecommender(ueritemtrain_mat, 32)
predicted_mat = model.svd_recommender_userblock()
#print (predicted_mat)
useritemtrain_mat = useritemtrain_df.to_numpy()
useritemtest_mat= useritemtest_df.to_numpy()
evaluation = Evaluation (useritemtrain_mat, useritemtest_mat, predicted_mat )
user_str = user_id_vec[0]

print (model.svd_recommender_user(user_str,user_id_vec,item_id_vec, 5))
print (evaluation.recallat_user_block())

