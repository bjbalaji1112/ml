#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Data Exploratory Analysis For Credit Card Data</h1>

# # Business Problem:

# <b>In order to effectively produce quality decisions in the modern credit card industry, knowledge must be gained through effective data analysis and modelling. Through the use of dynamic data-driven decision-making tools and procedures, information can be gathered to successfully evaluate all aspects of credit card operations. PSPD Bank has banking operations in more than 50 countries across the globe. Mr. Jim Watson, CEO, wants to evaluate areas of bankruptcy, fraud and collections, respond to customer requests for help with proactive offers and services. </b><br><br>
# <b>Following are some of Mr. Watson's questionsn to understand the customer spend and repayment behaviour</b>

# # Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# # Import the datasets

# In[2]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


customer = pd.read_csv("/kaggle/input/credit-card-exploratory-data-analysis/Customer Acqusition.csv",usecols=["Customer","Age","City","Product","Limit","Company","Segment"])
repay = pd.read_csv("/kaggle/input/credit-card-exploratory-data-analysis/Repayment.csv",usecols = ["Customer","Month","Amount"])
spend = pd.read_csv("/kaggle/input/credit-card-exploratory-data-analysis/spend.csv",usecols=["Customer","Month","Type","Amount"])


# In[4]:


customer.head()


# In[5]:


repay.head()


# In[6]:


spend.head(2)


# # Exploratory Data Analysis

# In[7]:


print(customer.shape)
print(repay.shape)
print(spend.shape)


# In[8]:


customer.dtypes


# In[9]:


repay.dtypes


# In[10]:


spend.dtypes


# In[11]:


spend.isnull().sum()


# In[12]:


customer.isnull().sum()


# In[13]:


repay.isnull().sum()


# In[14]:


# dropping null values present in 'repay' data set
repay.dropna(inplace=True)


# In[15]:


repay.isnull().sum()


# # (1) In the above dataset,

# ## (a) In case age is less than 18, replace it with mean of age values.

# In[16]:


mean_original = customer["Age"].mean()


# In[17]:


print("The mean of Age column is",mean_original)


# In[18]:


#replacing age less than 18 with mean of age values
customer.loc[customer["Age"] < 18,"Age"] = customer["Age"].mean()


# In[19]:


mean_new = customer["Age"].mean()


# In[20]:


print("The new mean of Age column is",mean_new)


# In[21]:


customer.loc[customer["Age"] < 18,"Age"]


# In[22]:


print("All the customers who have age less than 18 have been replaced by mean of the age column.")


# ##  (b) In case spend amount is more than the limit, replace it with 50% of that customer’s limit. (customer’s limit provided in acquisition table is the per transaction limit on his card)

# In[23]:


customer.head(2)


# In[24]:


spend.head(2)


# In[25]:


#merging customer and spend table on the basis of "Customer" column
customer_spend = pd.merge(left=customer,right=spend,on="Customer",how="inner")


# In[26]:


customer_spend.head()


# In[27]:


customer_spend.shape


# In[28]:


#all the customers whose spend amount is more than the limit,replacing with 50% of that customer’s limit
customer_spend[customer_spend["Amount"] > customer_spend['Limit']]


# In[29]:


#if customer's spend amount is more than the limit,replacing with 50% of that customer’s limit
customer_spend.loc[customer_spend["Amount"] > customer_spend["Limit"],"Amount"] = (50 * customer_spend["Limit"]).div(100)


# In[30]:


#there are no customers left whose spend amount is more than the limit
customer_spend[customer_spend["Amount"] > customer_spend['Limit']]


# ## (c)  Incase the repayment amount is more than the limit, replace the repayment with the limit.

# In[31]:


customer.head(1)


# In[32]:


repay.head(1)


# In[33]:


#merging customer and spend table on the basis of "Customer" column
customer_repay = pd.merge(left=repay,right=customer,on="Customer",how="inner")


# In[34]:


customer_repay.head()


# In[35]:


#all the customers where repayment amount is more than the limit.
customer_repay[customer_repay["Amount"] > customer_repay["Limit"]]


# In[36]:


#customers where repayment amount is more than the limit, replacing the repayment with the limit.
customer_repay.loc[customer_repay["Amount"] > customer_repay["Limit"],"Amount"] = customer_repay["Limit"]


# In[37]:


#there are no customers left where repayment amount is more than the limit.
customer_repay[customer_repay["Amount"] > customer_repay["Limit"]]


# # (2) From the above dataset create the following summaries:

# ## (a) How many distinct customers exist?

# In[38]:


distinct_customers = customer["Customer"].nunique()


# In[39]:


print("Number of distinct customers are",distinct_customers)


# ## (b) How many distinct categories exist?

# In[40]:


#customers from different segments
customer["Segment"].value_counts()


# In[41]:


plt.figure(figsize=(8,6))
sns.countplot('Segment',data=customer)
plt.show()


# In[42]:


print("We can see from the countplot that number of distinct categories are", len(customer["Segment"].value_counts()))


# ## (c) What is the average monthly spend by customers?

# In[43]:


spend.head()


# In[44]:


#converting Month column of "spend" table to date time format
spend['Month'] = pd.to_datetime(spend['Month'])


# In[45]:


spend.head()


# In[46]:


#creating new columns which show "Month" and "Year"
spend['Monthly'] = spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
spend['Yearly'] = spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))


# In[47]:


spend.head()


# In[48]:


#grouping the dataset based on 'Yearly' and 'monthly'
customer_spend_group= round(spend.groupby(['Yearly','Monthly']).mean(),2)


# In[49]:


customer_spend_group


# ##  (d) What is the average monthly repayment by customers?

# In[50]:


repay.head(2)


# In[51]:


#coverting "Month" column to date time format
repay["Month"] = pd.to_datetime(repay["Month"])


# In[52]:


repay.head(2)


# In[53]:


repay.dtypes


# In[54]:


#creating new columns which show "Month" and "Year"
repay['Monthly'] = repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
repay['Yearly'] = repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))


# In[55]:


#grouping the dataset based on 'Yearly' and 'monthly'
customer_repay_group= round(repay.groupby(['Yearly','Monthly']).mean(),2)


# In[56]:


customer_repay_group


# ## (e)  If the monthly rate of interest is 2.9%, what is the profit for the bank for each month? 

# In[57]:


#merging all the three tables. Alreaady merged customer and spend table in 'customer_spend'. Using "customer_spend" and "repay"
#table to form the final "customer_spend_repay" table
customer_spend_repay = pd.merge(left=customer_spend,right=repay,on="Customer",how="inner")


# In[58]:


customer_spend_repay.head(2)


# In[59]:


# renaming the columns for clearity
customer_spend_repay.rename(columns={"Amount_x":"Spend_Amount","Amount_y":"Repay_Amount"},inplace=True)


# In[60]:


customer_spend_repay.head()


# In[61]:


# grouping the data based on "Yearly","Month_x" columns to get the 'Spend_Amount'and 'Repay_Amount'
interest_group = customer_spend_repay.groupby(["Yearly","Monthly"])['Spend_Amount','Repay_Amount'].sum()


# In[62]:


interest_group


# In[63]:


# Monthly Profit = Monthly repayment – Monthly spend.
interest_group['Monthly Profit'] = interest_group['Repay_Amount'] - interest_group['Spend_Amount']


# In[64]:


interest_group


# In[65]:


#interest earned is 2.9% of Monthly Profit
interest_group['Interest Earned'] = (2.9* interest_group['Monthly Profit'])/100


# In[66]:


interest_group


# ## (f) What are the top 5 product types?

# In[67]:


spend.head()


# In[68]:


#top 5 product types on which customer is spending
spend['Type'].value_counts().head()


# In[69]:


spend['Type'].value_counts().head(5).plot(kind='bar')
plt.show()


# ## (g)  Which city is having maximum spend?

# In[70]:


customer_spend.head()


# In[71]:


city_spend = customer_spend.groupby("City")["Amount"].sum().sort_values(ascending=False)


# In[72]:


city_spend


# In[73]:


plt.figure(figsize=(5,10))
city_spend.plot(kind="pie",autopct="%1.0f%%",shadow=True,labeldistance=1.0,explode=[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
plt.title("Amount spent on credit card by customers from different cities")
plt.show()


# In[74]:


print("From above pie chart we can see that Cochin is having maximum spend.")


# ## (h) Which age group is spending more money?

# In[75]:


#creating new column "Age Group" with 8 bins between 18 to 88 
customer_spend["Age Group"] =  pd.cut(customer_spend["Age"],bins=np.arange(18,88,8),labels=["18-26","26-34", "34-42" ,"42-50" ,"50-58","58-66","66-74","74-82"],include_lowest=True)


# In[76]:


customer_spend.head()


# In[77]:


#grouping data based on "Age Group" and finding the amount spend by each age group and arranging in descending oreder
age_spend = customer_spend.groupby("Age Group")['Amount'].sum().sort_values(ascending=False)


# In[78]:


age_spend


# In[79]:


plt.figure(figsize=(5,10))
age_spend.plot(kind = "pie",autopct="%1.0f%%",explode=[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],shadow=True)
plt.show()


# In[80]:


print("From the pie chart shown above we can say that age group 42 - 50 is spending more money")


# ## (i) Who are the top 10 customers in terms of repayment?

# In[81]:


customer_repay.head()


# In[82]:


#grouping based on "Customer" column to find top 10 customers
customer_repay.groupby("Customer")[["Amount"]].sum().sort_values(by="Amount",ascending=False).head(10)


# ## (3) Calculate the city wise spend on each product on yearly basis. Also include a graphical representation for the same.

# In[83]:


customer_spend.head()


# In[84]:


#converting "Month" column to date time 
customer_spend["Month"] = pd.to_datetime(customer_spend["Month"])


# In[85]:


#creating new column "year" 
customer_spend['Year'] = customer_spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))


# In[86]:


customer_spend.head(2)


# In[87]:


customer_spend_pivot = pd.pivot_table(data = customer_spend,index=["City","Year"],columns='Product',aggfunc="sum",values="Amount")


# In[88]:


customer_spend_pivot


# In[89]:


customer_spend_pivot.plot(kind="bar",figsize=(18,5),width=0.8)
plt.ylabel("Spend Amount")
plt.title("Amount spended by customers according to year and city")
plt.show()


# # (4) Create graphs for
#  

# ## (a) Monthly comparison of total spends, city wise

# In[90]:


customer_spend.head()


# In[91]:


#creating new column "Monthly" 
customer_spend['Monthly'] = customer_spend['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))


# In[92]:


customer_spend.head()


# In[93]:


#grouping data based on "Monthly" and "City" columns
month_city = customer_spend.groupby(["Monthly","City"])[["Amount"]].sum().sort_index().reset_index()


# In[94]:


#creating pivot table based on "Monthly" and "City" columns
month_city =pd.pivot_table(data=customer_spend,values='Amount',index='City',columns='Monthly',aggfunc='sum')


# In[95]:


month_city


# In[96]:


month_city.plot(kind="bar",figsize=(18,6),width=0.8)
plt.show()


# ## (b) Comparison of yearly spend on air tickets

# In[97]:


customer_spend.head()


# In[98]:


air_tickets = customer_spend.groupby(["Year","Type"])[["Amount"]].sum().reset_index()


# In[99]:


filtered = air_tickets.loc[air_tickets["Type"]=="AIR TICKET"]


# In[100]:


filtered


# In[101]:


plt.bar(filtered["Year"],height=filtered["Amount"],color="orange")
plt.xlabel("Year")
plt.ylabel("Amount Spent")
plt.title("Comparison of yearly spend on air tickets")
plt.show()


# ## (c)  Comparison of monthly spend for each product (look for any seasonality that exists in terms of spend)

# In[102]:


customer_spend.head(2)


# In[103]:


#creating pivot table based on "Monthly" and "Product" columns
product_wise = pd.pivot_table(data=customer_spend,index='Product',columns='Monthly',values='Amount',aggfunc='sum')


# In[104]:


product_wise


# In[105]:


product_wise.plot(kind="bar",figsize=(18,6),width=0.8)
plt.ylabel("Amount Spend")
plt.title("Amount spent monthly on different products")
plt.show()


# <b>We can see from the above graph that the sales are high for all the Products during the months:</b>
# <ul>
#     <li>January</li>
#     <li>February</li>
#     <li>March</li>
#     <li>April</li>
#     <li>May</li></ul>
# <b> Out of these months,highest sales are in January </b> 

# ## (5) Write user defined PYTHON function to perform the following analysis: You need to find top 10 customers for each city in terms of their repayment amount by different products and by different time periods i.e. year or month. The user should be able to specify the product (Gold/Silver/Platinum) and time period (yearly or monthly) and the function should automatically take these inputs while identifying the top 10 customers.

# In[106]:


customer_repay.head(2)


# In[107]:


# converting 'Month' column to date time format
customer_repay['Month'] = pd.to_datetime(customer_repay['Month'])


# In[108]:


#creating new column "Monthly" and "Yearly" using already existing 'Month' column
customer_repay['Monthly'] = customer_repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
customer_repay['Yearly'] = customer_repay['Month'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))


# In[109]:


def summary_report(product,timeperiod):
    print('Give the product name and timeperiod for which you want the data')
    if product.lower()=='gold' and timeperiod.lower()=='monthly':
        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='gold' and timeperiod.lower()=='yearly':
        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Gold',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='silver' and timeperiod.lower()=='monthly':
        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='silver' and timeperiod.lower()=='yearly':
        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Silver',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    if product.lower()=='platinum' and timeperiod.lower()=='monthly':
        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Monthly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Platinum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    elif product.lower()=='platinum' and timeperiod.lower()=='yearly':
        pivot = customer_repay.pivot_table(index=['Product','City','Customer'],columns='Yearly',aggfunc='sum',values='Amount')
        result = pivot.loc[('Platinum',['BANGALORE','COCHIN','CALCUTTA','BOMBAY','CHENNAI','TRIVANDRUM','PATNA','DELHI']),:]
    return result


# In[110]:


summary_report('gold','monthly')


# In[ ]:




