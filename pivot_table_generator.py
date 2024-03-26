#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


users=pd.read_csv('Users.csv',encoding='latin1')
users.head()


# In[3]:


books=pd.read_csv('Books.csv',encoding='ISO-8859-1',low_memory=False)
books.head()


# In[4]:


ratings = pd.read_csv('Ratings.csv', encoding='latin1')
ratings.head()


# In[5]:


users.shape


# In[6]:


books.shape


# In[7]:


ratings.shape


# In[8]:


users.duplicated().sum()


# In[9]:


books.duplicated().sum()


# In[10]:


ratings.duplicated().sum()


# In[11]:


def unique_values(dataset):
    list_unique_valeus = [dataset[col].nunique() for col in dataset.columns]
    list_cols = dataset.columns.tolist()
    data = list(zip(list_cols,list_unique_valeus))
    df = pd.DataFrame(data,columns = ['Column','No of Unique Values'])
    return df


# In[12]:


unique_values(users)


# In[13]:


unique_values(ratings)


# In[14]:


unique_values(books)


# In[15]:


merged_df=pd.merge(users,ratings,on='User-ID')
merged_df=pd.merge(merged_df,books,on='ISBN')


# In[16]:


merged_df.head()


# In[17]:


merged_df.shape


# In[18]:


merged_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)


# In[19]:


merged_df.columns= merged_df.columns.str.replace('-', '_')


# In[20]:


merged_df.columns


# In[21]:


def missing_values(df):
    miss = df.isnull().sum()
    per = df.isnull().mean()
    df = pd.concat([miss,per*100],keys = ['Missing_Values','Percentage'], axis = 1)
    return df


# In[22]:


missing_values(merged_df)


# In[23]:


merged_df.head()


# In[24]:


merged_df.loc[(merged_df.Age > 100) | (merged_df.Age < 5), 'Age'] = np.nan


# In[25]:


merged_df.isna().sum()


# In[26]:


# Create a country column at the place of location
merged_df['Country'] = merged_df['Location'].astype(str).apply(lambda x:x.split(',')[-1])
# Drop the location column
merged_df.drop('Location',axis=1,inplace=True) # inplace =True means we are changing original datafram itself


# In[27]:


merged_df['Country'].unique()


# In[28]:


merged_df['Country'] = merged_df['Country'].replace(' ','other').replace(' n/a','other')


# In[29]:


merged_df['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[30]:


def missing_values(df):
    miss = df.isnull().sum()
    per = df.isnull().mean()
    df = pd.concat([miss,per*100],keys = ['Missing_Values','Percentage'], axis = 1)
    return df


# In[31]:


missing_values(merged_df)


# In[32]:


merged_df.head()


# In[33]:


merged_df['Year_Of_Publication'].unique()


# In[34]:


merged_df[merged_df['Publisher'].isnull()]
# merged_df.loc[(merged_df['publisher'].isnull()),:]


# In[35]:


missing_values(merged_df)


# In[36]:


merged_df.loc[(merged_df.Age > 100) | (merged_df.Age < 5), 'Age'] = np.nan


# In[37]:


# finding outlier in age
# Box plot for age
sns.boxplot(merged_df['Age']);
plt.title('Find outlier data in Age column')


# In[38]:


merged_df.loc[(merged_df.Age > 100) | (merged_df.Age < 5), 'Age'] = np.nan


# In[39]:


merged_df.isna().sum()


# In[40]:


merged_df['Age'] = merged_df['Age'].fillna(merged_df.groupby('Country')['Age'].transform('median'))


# In[41]:


merged_df.isna().sum()


# In[42]:


merged_df['Age'].fillna(merged_df.Age.mean(),inplace=True)


# In[43]:


merged_df.isna().sum()


# In[44]:


merged_df.loc[merged_df['Year_Of_Publication'] == 'DK Publishing Inc',:]


# In[45]:


merged_df.loc[merged_df['Year_Of_Publication'] == 'Gallimard',:]


# In[46]:


merged_df.loc[merged_df.ISBN == '2070426769','Year_Of_Publication'] = 2003
merged_df.loc[merged_df.ISBN == '2070426769','Book_Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
merged_df.loc[merged_df.ISBN == '2070426769','Publisher'] = "Gallimard"
merged_df.loc[merged_df.ISBN == '2070426769','Book_Title'] = "Peuple du ciel, suivi de 'Les Bergers"
merged_df.loc[merged_df.ISBN == '2070426769',:]


# In[47]:


merged_df.loc[merged_df.ISBN == '2070426769','Year_Of_Publication'] = 2003
merged_df.loc[merged_df.ISBN == '2070426769','Book_Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
merged_df.loc[merged_df.ISBN == '2070426769','Publisher'] = "Gallimard"
merged_df.loc[merged_df.ISBN == '2070426769','Book_Title'] = "Peuple du ciel, suivi de 'Les Bergers"
merged_df.loc[merged_df.ISBN == '2070426769',:]


# In[48]:


merged_df['Year_Of_Publication']=pd.to_numeric(merged_df['Year_Of_Publication'], errors='coerce')
merged_df['Year_Of_Publication'].unique()


# In[49]:


merged_df.loc[(merged_df['Year_Of_Publication'] > 2006) | (merged_df['Year_Of_Publication'] == 0),'Year_Of_Publication'] = np.NAN
merged_df['Year_Of_Publication'].fillna(round(merged_df['Year_Of_Publication'].median()), inplace=True)


# In[50]:


merged_df.isna().sum()


# In[51]:


merged_df.loc[merged_df.Publisher.isnull(),:]


# In[52]:


merged_df.Publisher.fillna('other',inplace=True)


# In[53]:


merged_df.loc[merged_df['Book_Author'].isnull(),:]


# In[54]:


merged_df['Book_Author'].fillna('other',inplace=True)


# In[55]:


merged_df.isna().sum()


# In[56]:


missing_values(merged_df)


# In[57]:


merged_df.shape


# In[58]:


u = merged_df.Age.value_counts().sort_index()
plt.bar(u.index, u.values)
plt.xlabel('Age')
plt.ylabel('Count of Users')
plt.xlim(xmin = 0)
plt.show()


# In[59]:


plt.figure(figsize=(15,7))
sns.countplot(y='Publisher',data=merged_df,order=pd.value_counts(books['Publisher']).iloc[:10].index)
plt.title('Top 10 Publishers')


# In[60]:


plt.figure(figsize=(15,7))
sns.countplot(y='Book_Author',data=merged_df,order=pd.value_counts(merged_df['Book_Author']).iloc[:10].index)
plt.title('Top 10 Authors')


# In[61]:


sns.distplot(merged_df.Age)
plt.title('Age Distribution Plot')


# In[62]:


plt.figure(figsize=(15,7))
sns.countplot(y='Country',data=merged_df,order=pd.value_counts(merged_df['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# In[63]:


# Pie Graph of top five countires.
palette_color = sns.color_palette('pastel')
explode = (0.1, 0, 0, 0, 0)
merged_df.Country.value_counts().iloc[:5].plot(kind='pie', colors=palette_color, autopct='%.0f%%', explode=explode, shadow=True)
plt.title('Top 5 countries', fontweight='bold');


# In[64]:


book_rating = merged_df.groupby(['Book_Title','Book_Author'])['Book_Rating'].agg(['count','mean']).sort_values(by='mean', ascending=False).reset_index()
sns.catplot(x='mean', y='Book_Title', data=book_rating[book_rating['count']>500][:10], kind='bar', palette = 'Paired',hue='Book_Author' )
plt.xlabel('Average Ratings')
plt.ylabel('Books')
plt.title('Most Famous Books', fontweight='bold');


# In[65]:


sns.barplot(x = merged_df['Book_Rating'].value_counts().index,y = merged_df['Book_Rating'].value_counts().values,palette = 'magma').set(title="Ratings Distribution", xlabel = "Rating",ylabel = 'Number of books')
plt.show();


# In[66]:


sns.countplot(x="Book_Rating",palette='Paired',data=merged_df)
plt.title("Ratings",fontweight='bold');


# In[67]:


merged_df.info()


# In[68]:


merged_df = merged_df.dropna(subset=['Book_Rating'])


# In[70]:


merged_df['Avg_Ratings'] =  merged_df.groupby('Book_Title')['Book_Rating'].transform('mean')


# In[71]:


merged_df['No_Of_Ratings'] = merged_df.groupby('Book_Title')['Book_Rating'].transform('count')


# In[72]:


popular_df = merged_df[['Book_Title','Avg_Ratings','No_Of_Ratings']]


# In[73]:


popular_df.drop_duplicates('Book_Title',inplace=True)


# In[74]:


popular_df.head()


# In[75]:


popular_df = popular_df[popular_df['No_Of_Ratings']>200].sort_values('Avg_Ratings',ascending=False)


# In[76]:


popular_df.head(10)


# In[77]:


merged_df.shape


# In[78]:


merged_df.columns


# In[79]:


x = merged_df.groupby('User_ID').count()['Book_Rating'] > 180


# In[80]:


x[x]


# In[81]:


merged_df['User_ID'].isin(x[x].index)


# In[82]:


print("Shape of merged dataframe : ",merged_df.shape)


# In[83]:


merged_df = merged_df[merged_df['Book_Rating']!=0]


# In[84]:


print("Shape of merged new dataframe : ",merged_df.shape)


# In[85]:


x = merged_df.groupby('User_ID').count()['Book_Rating'] >180
filtered_df = merged_df[merged_df['User_ID'].isin(x[x].index)]


# In[86]:


y = merged_df.groupby('Book_Title').count()['Book_Rating'] >50
filtered_df = filtered_df[filtered_df['Book_Title'].isin(y[y].index)]


# In[87]:


filtered_df.shape


# In[88]:


filtered_df.head()


# In[89]:


pt = filtered_df.pivot_table(index='Book_Title',columns='User_ID',values='Book_Rating').fillna(0)
pt


# In[90]:


df = pd.DataFrame(filtered_df)


# In[99]:


df.to_csv('output_file.csv', index=False)


# In[ ]:




