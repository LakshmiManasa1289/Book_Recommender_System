#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import scipy
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from scipy.sparse.linalg import svds


# # Reading data

# In[7]:


users=pd.read_csv('Users.csv',encoding='latin1')
users.head()


# In[8]:


books=pd.read_csv('Books.csv',encoding='ISO-8859-1')
books.head()


# In[9]:


ratings = pd.read_csv('Ratings.csv', encoding='latin1', error_bad_lines=False)
ratings.head()


# In[10]:


users.shape


# In[11]:


books.shape


# In[12]:


ratings.shape


# In[13]:


users.info()


# In[14]:


books.info()


# In[15]:


ratings.info()


# In[16]:


users.duplicated().sum()


# In[17]:


books.duplicated().sum()


# In[18]:


ratings.duplicated().sum()


# In[19]:


# Dataset Columns
print(f'Columns in Users: {users.columns}')
print(f'Columns in Books: {books.columns}')
print(f'Columns in Ratings: {ratings.columns}')


# In[20]:


def unique_values(dataset):
    list_unique_valeus = [dataset[col].nunique() for col in dataset.columns]
    list_cols = dataset.columns.tolist()
    data = list(zip(list_cols,list_unique_valeus))
    df = pd.DataFrame(data,columns = ['Column','No of Unique Values'])
    return df


# In[21]:


unique_values(users)


# In[22]:


unique_values(ratings)


# In[23]:


unique_values(books)


# # Data Pre-Processing and Cleaning

# In[24]:


merged_df=pd.merge(users,ratings,on='User-ID')
merged_df=pd.merge(merged_df,books,on='ISBN')


# In[25]:


merged_df.head()


# In[26]:


merged_df.columns


# In[27]:


merged_df.info()


# In[28]:


merged_df.shape


# In[29]:


merged_df.duplicated().sum()


# In[30]:


merged_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# In[31]:


merged_df.columns= merged_df.columns.str.replace('-', '_')


# In[32]:


merged_df.columns


# In[33]:


# Create a country column at the place of location
merged_df['Country'] = merged_df['Location'].astype(str).apply(lambda x:x.split(',')[-1])
# Drop the location column
merged_df.drop('Location',axis=1,inplace=True) # inplace =True means we are changing original datafram itself


# In[34]:


merged_df['Country'].unique()


# In[35]:


merged_df['Country'] = merged_df['Country'].replace(' ','other').replace(' n/a','other')


# In[36]:


merged_df['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[37]:


def missing_values(df):
    miss = df.isnull().sum()
    per = df.isnull().mean()
    df = pd.concat([miss,per*100],keys = ['Missing_Values','Percentage'], axis = 1)
    return df


# In[38]:


missing_values(merged_df)


# In[39]:


merged_df.head()


# In[40]:


merged_df['Year_Of_Publication'].unique()


# In[41]:


merged_df[merged_df['Publisher'].isnull()]
# merged_df.loc[(merged_df['publisher'].isnull()),:]


# In[42]:


missing_values(merged_df)


# # Exploratory Data Analysis

# In[43]:


merged_df.corr()


# In[44]:


# finding outlier in age
# Box plot for age
sns.boxplot(merged_df['Age']);
plt.title('Find outlier data in Age column')


# In[45]:


merged_df.loc[(merged_df.Age > 100) | (merged_df.Age < 5), 'Age'] = np.nan


# In[46]:


merged_df.isna().sum()


# In[47]:


merged_df['Age'] = merged_df['Age'].fillna(merged_df.groupby('Country')['Age'].transform('median'))


# In[48]:


merged_df.isna().sum()


# In[49]:


merged_df['Age'].fillna(merged_df.Age.mean(),inplace=True)


# In[50]:


merged_df.isna().sum()


# In[51]:


merged_df.loc[merged_df['Year_Of_Publication'] == 'DK Publishing Inc',:]



# In[52]:


merged_df.loc[merged_df.ISBN == '0789466953','Year_Of_Publication'] = 2000
merged_df.loc[merged_df.ISBN == '0789466953','Book_Author'] = "James Buckley"
merged_df.loc[merged_df.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
merged_df.loc[merged_df.ISBN == '0789466953','Book_Title'] = "DK Readers: Creating the X-Men, How Comic merged_df Come to Life (Level 4: Proficient Readers)"
#ISBN '078946697X'
merged_df.loc[merged_df.ISBN == '078946697X','Year_Of_Publication'] = 2000
merged_df.loc[merged_df.ISBN == '078946697X','Book_Author'] = "Michael Teitelbaum"
merged_df.loc[merged_df.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
merged_df.loc[merged_df.ISBN == '078946697X','Book_Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
merged_df.loc[(merged_df.ISBN == '0789466953') | (merged_df.ISBN == '078946697X'),:]


# In[53]:


merged_df.loc[merged_df['Year_Of_Publication'] == 'Gallimard',:]


# In[54]:


merged_df.loc[merged_df.ISBN == '2070426769','Year_Of_Publication'] = 2003
merged_df.loc[merged_df.ISBN == '2070426769','Book_Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
merged_df.loc[merged_df.ISBN == '2070426769','Publisher'] = "Gallimard"
merged_df.loc[merged_df.ISBN == '2070426769','Book_Title'] = "Peuple du ciel, suivi de 'Les Bergers"
merged_df.loc[merged_df.ISBN == '2070426769',:]


# In[55]:


merged_df['Year_Of_Publication']=pd.to_numeric(merged_df['Year_Of_Publication'], errors='coerce')
merged_df['Year_Of_Publication'].unique()


# In[56]:


merged_df.loc[(merged_df['Year_Of_Publication'] > 2006) | (merged_df['Year_Of_Publication'] == 0),'Year_Of_Publication'] = np.NAN
merged_df['Year_Of_Publication'].fillna(round(merged_df['Year_Of_Publication'].median()), inplace=True)


# In[57]:


merged_df.isna().sum()


# In[58]:


merged_df.loc[merged_df.Publisher.isnull(),:]


# In[59]:


merged_df.Publisher.fillna('other',inplace=True)


# In[60]:


merged_df.loc[merged_df['Book_Author'].isnull(),:]



# In[61]:


merged_df['Book_Author'].fillna('other',inplace=True)


# In[62]:


merged_df.isna().sum()


# In[63]:


missing_values(merged_df)


# In[64]:


merged_df.shape


# In[65]:


u = merged_df.Age.value_counts().sort_index()
plt.bar(u.index, u.values)
plt.xlabel('Age')
plt.ylabel('Count of Users')
plt.xlim(xmin = 0)
plt.show()


# In[66]:


plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=merged_df,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[67]:


plt.figure(figsize=(15,7))
sns.countplot(y='Book_Author',data=merged_df,order=pd.value_counts(merged_df['Book_Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[68]:


sns.distplot(merged_df.Age)
plt.title('Age Distribution Plot')


# In[69]:


plt.figure(figsize=(15,7))
sns.countplot(y='Country',data=merged_df,order=pd.value_counts(merged_df['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# In[70]:


# Pie Graph of top five countires.
palette_color = sns.color_palette('pastel')
explode = (0.1, 0, 0, 0, 0)
merged_df.Country.value_counts().iloc[:5].plot(kind='pie', colors=palette_color, autopct='%.0f%%', explode=explode, shadow=True)
plt.title('Top 5 countries', fontweight='bold');


# In[71]:


book_rating = merged_df.groupby(['Book_Title','Book_Author'])['Book_Rating'].agg(['count','mean']).sort_values(by='mean', ascending=False).reset_index()
sns.catplot(x='mean', y='Book_Title', data=book_rating[book_rating['count']>500][:10], kind='bar', palette = 'Paired',hue='Book_Author' )
plt.xlabel('Average Ratings')
plt.ylabel('Books')
plt.title('Most Famous Books', fontweight='bold');


# In[72]:


sns.barplot(x = merged_df['Book_Rating'].value_counts().index,y = merged_df['Book_Rating'].value_counts().values,palette = 'magma').set(title="Ratings Distribution", xlabel = "Rating",ylabel = 'Number of books')
plt.show();


# In[73]:


sns.countplot(x="Book_Rating",palette='Paired',data=merged_df)
plt.title("Ratings",fontweight='bold');


# In[74]:


merged_df.corr()


# In[75]:


plt.figure(figsize=(14,10))
sns.heatmap(merged_df.corr(),annot=True,cmap='terrain')


# # Popularity Based Recommender System

# In[76]:


merged_df['Avg_Ratings'] =  merged_df.groupby('Book_Title')['Book_Rating'].transform('mean')


# In[77]:


merged_df['No_Of_Ratings'] = merged_df.groupby('Book_Title')['Book_Rating'].transform('count')


# In[78]:


popular_df = merged_df[['Book_Title','Avg_Ratings','No_Of_Ratings']]


# In[79]:


popular_df.drop_duplicates('Book_Title',inplace=True)


# In[80]:


popular_df.head()


# In[81]:


popular_df = popular_df[popular_df['No_Of_Ratings']>200].sort_values('Avg_Ratings',ascending=False)


# In[82]:


popular_df.head(10)


# # Collaborative Filtering

# # Item Based

# In[83]:


merged_df.shape


# In[84]:


merged_df.columns


# In[85]:


x = merged_df.groupby('User_ID').count()['Book_Rating'] > 180


# In[86]:


x[x]


# In[87]:


merged_df['User_ID'].isin(x[x].index)


# In[88]:


print("Shape of merged dataframe : ",merged_df.shape)


# In[89]:


merged_df = merged_df[merged_df['Book_Rating']!=0]


# In[90]:


print("Shape of merged new dataframe : ",merged_df.shape)


# In[91]:


x = merged_df.groupby('User_ID').count()['Book_Rating'] >180
filtered_df = merged_df[merged_df['User_ID'].isin(x[x].index)]


# In[92]:


y = merged_df.groupby('Book_Title').count()['Book_Rating'] >50
filtered_df = filtered_df[filtered_df['Book_Title'].isin(y[y].index)]


# In[93]:


filtered_df.shape


# In[94]:


filtered_df.head()


# In[95]:


pt = filtered_df.pivot_table(index='Book_Title',columns='User_ID',values='Book_Rating').fillna(0)
pt


# In[96]:


similarity_scores_books = cosine_similarity(pt)


# In[97]:


similarity_scores_books


# In[98]:


similarity_scores=cosine_similarity(pt)


# In[99]:


for i,j in enumerate([1,2,3]):
    print(f"Index : {i} value {j}")


# In[100]:


def recommend_book(book_name):
  """
  Description: It takes a book name and return data frame with similarity score
  Function: recommend_book
  Argument: book_name
  Return type : dataframe
  """
  index = np.where(pt.index == book_name)[0][0] # finding index of same book
  similar_books = sorted(list(enumerate(similarity_scores[index])), key = lambda x:x[1], reverse = True)[1:6] # creating the list tuple of index with respect to similarity score

  # print(similar_books)

  print("\n----------------Recommended books-----------------\n")
  for i in similar_books:
    print(pt.index[i[0]])
  print("\n.....................................................\n")
  return find_similarity_score(similar_books,pt)


# In[101]:


def find_similarity_score(similarity_scores,pivot_table):

  """
  Description: It takes similarity_Score and pivot table and return dataframe.
  function : find_similarity_Score
  Output : dataframe
  Argument  similarity_score and pivot table
  """
  list_book = []
  list_sim = []
  for i in similarity_scores:
    index_ = i[0]
    sim_ = i[1]
    list_sim.append(sim_)
    # list_book.append(pivot_table[pivot_table.index == index_]['Book-Title'][index_])
    list_book.append(pivot_table.iloc[index_,:].name)

    df = pd.DataFrame(list(zip(list_book, list_sim)),
               columns =['Book', 'Similarity'])
  # df =pd.DataFrame([list_book, list_sim], columns = ["Book",'Similarity_Score'])
  return df


# In[102]:


recommend_book('Harry Potter and the Prisoner of Azkaban (Book 3)')


# In[103]:


recommend_book('Harry Potter and the Prisoner of Azkaban (Book 3)')


# In[104]:


df_matrix= csr_matrix(pt.values)
df_matrix


# # Building KNN Model

# In[105]:


knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors=5)
knn.fit(pt)


# In[106]:


def recommend_book(book, n_values=11):

  distances, indices = knn.kneighbors(pt.loc[book,:].values.reshape(1, -1), n_neighbors = n_values)
  dist = distances.flatten().tolist()
  books = []
  for i in range(1, len(indices.flatten())):
    books.append(pt.index[indices.flatten()[i]])

  data = list(zip(book,dist))
  df = pd.DataFrame(data,columns=['book','Distance'])
  return df


# In[107]:


recommend_book("Harry Potter and the Sorcerer's Stone (Book 1)")


# # User Based

# In[108]:


users_ratings_count_df = merged_df.groupby(['Book_Title', 'User_ID']).size().groupby('User_ID').size()
print('Number of users: %d' % len(users_ratings_count_df))
users_with_enough_ratings_df = users_ratings_count_df[users_ratings_count_df >50].reset_index()[['User_ID']] # Users who rated more than 50 books
print('Number of users with at least 10 ratings: %d' % len(users_with_enough_ratings_df))


# In[109]:


print('Number of ratings : %d' % len(merged_df))
ratings_from_selected_users_df = merged_df.merge(users_with_enough_ratings_df)
print('Number of ratings from users with at least 100 interactions: %d' % len(ratings_from_selected_users_df))


# In[110]:


ratings_from_selected_users_df.head()


# In[111]:


le = preprocessing.LabelEncoder()
le.fit(merged_df['Book_Title'].unique())


# In[112]:


def smooth_user_preference(x):
    return math.log(1+x, 2)
ratings_full_df = ratings_from_selected_users_df.groupby(['Book_Title','User_ID'])['Book_Rating'].sum().apply(smooth_user_preference).reset_index()
print('Number of unique user/item interactions: %d' % len(ratings_full_df))
ratings_full_df.head()


# In[113]:


ratings_train_df, ratings_test_df = train_test_split(ratings_full_df,
                                   test_size=0.20,
                                   stratify=ratings_full_df['User_ID'],
                                   random_state=42)

print('Number of ratings on Train set: %d' % len(ratings_train_df))
print('Number of ratings on Test set: %d' % len(ratings_test_df))


# In[114]:


users_items_pivot_matrix_df = ratings_train_df.pivot(index='User_ID', columns='Book_Title', values= 'Book_Rating').fillna(0)
users_items_pivot_matrix_df.head()


# In[115]:


users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_items_pivot_matrix[:10]


# In[116]:


users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]


# # Single Value Decomposition

# In[117]:


NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)


# In[118]:


users_items_pivot_matrix.shape


# In[119]:


U.shape


# In[120]:


sigma = np.diag(sigma)
sigma.shape


# In[121]:


Vt.shape


# In[122]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings


# In[123]:


all_user_predicted_ratings.shape


# In[124]:


cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head()


# In[125]:


len(cf_preds_df.columns)


# In[126]:


get_ipython().system('pip install scikit-surprise')


# In[127]:


from surprise import SVD
from surprise import SVDpp,accuracy
from surprise import NMF
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


# In[128]:


minimum_rating = min(ratings_full_df['Book_Rating'].values)




maximum_rating = max(ratings_full_df['Book_Rating'].values)


# In[129]:


reader = Reader(rating_scale=(minimum_rating,maximum_rating))

data = Dataset.load_from_df(ratings_full_df[['User_ID','Book_Title', 'Book_Rating']], reader)


# In[130]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[131]:


ratings_full_df.head(1)


# In[132]:


user_id = '96448'

book_rating = '3.321928'

prediction = algo.predict(uid=user_id, iid=book_rating)

print("Predicted rating of user with id {} for book with id {}: {}".format(user_id, book_rating, round(prediction.est,3)))






# In[133]:


# Predictions- actual and estimated
#predictions


# In[134]:


#SVDpp


# In[135]:


#Train test build  and model building


# In[136]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = SVDpp()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[137]:


ratings_full_df.head(1)


# In[138]:


user_id = '96448'

book_rating = '3.321928'

prediction = algo.predict(uid=user_id, iid=book_rating)

print("Predicted rating of user with id {} for book with id {}: {}".format(user_id, book_rating, round(prediction.est,3)))


# In[139]:


#predictions


# In[140]:


#NMF
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = NMF()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE


# In[141]:


ratings_full_df.head(1)


# In[142]:


#predictions


# In[143]:


from surprise import SlopeOne, CoClustering


# In[144]:


# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=0.25)

# We'll use the famous SVD algorithm
algo = SlopeOne()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Then compute MAE
accuracy.mae(predictions)


# In[145]:


ratings_full_df.head(1)


# In[146]:


#predictions


# In[152]:


from surprise.model_selection import cross_validate
from surprise import SVD, SVDpp, NMF
import matplotlib


# In[153]:


svd = cross_validate(SVD(), data, cv=5, n_jobs=-1, verbose=False)

svdpp = cross_validate(SVDpp(), data, cv=5, n_jobs=-1, verbose=False)

nmf = cross_validate(NMF(), data, cv=5, n_jobs=-1, verbose=False)

slope = cross_validate(SlopeOne(), data, cv=5, n_jobs=-1, verbose=False)


df_results = pd.DataFrame(columns=['Method', 'RMSE', 'MAE'])

df_results.loc[len(df_results)]=['SVD', round(svd['test_rmse'].mean(),5),round(svd['test_mae'].mean(),5)]

df_results.loc[len(df_results)]=['SVD++', round(svdpp['test_rmse'].mean(),5),round(svdpp['test_mae'].mean(),5)]

df_results.loc[len(df_results)]=['NMF', round(nmf['test_rmse'].mean(),5),round(nmf['test_mae'].mean(),5)]

df_results.loc[len(df_results)]=['SlopeOne', round(slope['test_rmse'].mean(),5),round(slope['test_mae'].mean(),5)]

display(df_results)


ax = df_results[['RMSE','MAE']].plot(kind='bar', figsize=(15,8))

ax.set_xticklabels(df_results['Method'].values)

ax.set_title('RMSE and MAE of different collaborative filtering algorithms')

plt.xticks(rotation=45)

matplotlib.rcParams.update({'font.size': 14})

plt.show();


# In[ ]:




