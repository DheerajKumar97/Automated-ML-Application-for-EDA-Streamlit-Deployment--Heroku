import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr

# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
from PIL import Image
image = Image.open('cover.jpg')
matplotlib.use("Agg")

def show_dtypes(x):
    return x.dtypes

def show_columns(x):
    return x.columns

def Show_Missing(x):
    return x.isna().sum()
def Show_Missing1(x):
    return x.isna().sum()

def Show_Missing2(x):
    return x.isna().sum()

def show_hist(x):
    return x.hist()

from scipy import stats
def Tabulation(x):
    table = pd.DataFrame(x.dtypes,columns=['dtypes'])
    table1 =pd.DataFrame(x.columns,columns=['Names'])
    table = table.reset_index()
    table= table.rename(columns={'index':'Name'})
    table['No of Missing'] = x.isnull().sum().values    
    table['No of Uniques'] = x.nunique().values
    table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
    table['First Observation'] = x.loc[0].values
    table['Second Observation'] = x.loc[1].values
    table['Third Observation'] = x.loc[2].values
    for name in table['Name'].value_counts().index:
        table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
    return table


def Numerical_variables(x):
    Num_var = [var for var in x.columns if x[var].dtypes!="object"]
    Num_var = x[Num_var]
    return Num_var

def categorical_variables(x):
    cat_var = [var for var in x.columns if x[var].dtypes=="object"]
    cat_var = x[cat_var]
    return cat_var

def impute(x):
    df=x.dropna()
    return df

def imputee(x):
    df=x.dropna()
    return df

def Show_pearsonr(x,y):
    result = pearsonr(x,y)
    return result

from scipy.stats import spearmanr
def Show_spearmanr(x,y):
    result = spearmanr(x,y)
    return result

import plotly.express as px
def plotly(a,x,y):
    fig = px.scatter(a, x=x, y=y)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def show_displot(x):
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(x)


        plt.subplot(122)
        x.plot.box(figsize=(16,5))

        plt.show()

def Show_DisPlot(x):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,7))
    return sns.distplot(x, bins = 25)

def Show_CountPlot(x):
    fig_dims = (18, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    return sns.countplot(x,ax=ax)

def plotly_histogram(a,x,y):
    fig = px.histogram(a, x=x, y=y)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def plotly_violin(a,x,y):
    fig = px.violin(a, x=x, y=y)
    fig.update_traces(marker=dict(size=10,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

def Show_PairPlot(x):
    return sns.pairplot(x)

def Show_HeatMap(x):
    f,ax = plt.subplots(figsize=(15, 15))
    return sns.heatmap(x.corr(),annot=True,ax=ax);

def label(x):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    x=le.fit_transform(x)
    return x

def label1(x):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    x=le.fit_transform(x)
    return x
   
def concat(x,y,z,axis):
    return pd.concat([x,y,z],axis)

def dummy(x):
    return pd.get_dummies(x)

import statsmodels.api as sm 
def qqplot(x):
    sm.qqplot(x, line ='45')

from scipy.stats import anderson
def Anderson_test(a):
    return anderson(a)

def PCA(x,y):
    from sklearn.decomposition import PCA
    pca =PCA(n_components=y)
    principlecomponents = pca.fit_transform(x)
    principledf = pd.DataFrame(data = principlecomponents)
    return principledf

def outlier(x):
    high=0
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    iqr = q3-q1
    low = q1-1.5*iqr
    high += q3+1.5*iqr
    outlier = (x.loc[(x < low) | (x > high)])
    return(outlier)

from scipy.stats import chi2_contingency,chi2

def check_cat_relation(x,y,confidence_interval):
    cross_table = pd.crosstab(x,y,margins=True)
    stat,p,dof,expected = chi2_contingency(cross_table)
    print("Chi_Square Value = {0}".format(stat))
    print("P-Value = {0}".format(p))
    alpha = 1 - confidence_interval
    return p,alpha
    if p > alpha:
        print(">> Accepting Null Hypothesis <<")
        print("There Is No Relationship Between Two Variables")
    else:
        print(">> Rejecting Null Hypothesis <<")
        print("There Is A Significance Relationship Between Two Variables")

from wordcloud import WordCloud
def wordcloud(x):
    wordcloud = WordCloud(width = 1000, height = 500).generate(" ".join(x))
    plt.imshow(wordcloud)
    plt.axis("off")
    return wordcloud

st.image(image, use_column_width=True)   
def main():
	st.title("Machine Learning Application for Automated EDA")
	
	st.info("This Web Application is created and maintained by *_DHEERAJ_ _KUMAR_ _K_*")
	"""https://github.com/DheerajKumar97""" 
	activities = ["General EDA","EDA For Linear Models","Feature Engineering","Model Building [IN Processs]"]	
	choice = st.sidebar.selectbox("Select Activities",activities)


	if choice == 'General EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")
			

			if st.checkbox("Show dtypes"):
				st.write(show_dtypes(df))

			if st.checkbox("Show Columns"):
				st.write(show_columns(df))

			if st.checkbox("Show Missing"):
				st.write(Show_Missing1(df))

			if st.checkbox("Aggregation Tabulation"):
				st.write(Tabulation(df))	

# 			if st.checkbox("Show Selected Columns"):
# 				selected_columns = st.multiselect("Select Columns",all_columns)
# 				new_df = df[selected_columns]
# 				st.dataframe(new_df)

                
			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",show_columns(df))
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Numerical Variables"):
				num_df = Numerical_variables(df)
				numer_df=pd.DataFrame(num_df)                
				st.dataframe(numer_df)

			if st.checkbox("Categorical Variables"):
				new_df = categorical_variables(df)
				catego_df=pd.DataFrame(new_df)                
				st.dataframe(catego_df)

			if st.checkbox("DropNA"):
				imp_df = impute(num_df)
				st.dataframe(imp_df)


			if st.checkbox("Missing after DropNA"):
				st.write(Show_Missing(imp_df))
               

			all_columns_names = show_columns(df)
			all_columns_names1 = show_columns(df)            
			selected_columns_names = st.selectbox("Select Columns Cross Tb",all_columns_names)
			selected_columns_names1 = st.selectbox("Select Columns Cross",all_columns_names1)
			if st.button("Generate Cross Tab"):
				st.dataframe(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))


			all_columns_names3 = show_columns(df)
			all_columns_names4 = show_columns(df)            
			selected_columns_name3 = st.selectbox("Select Columns Pearsonr",all_columns_names3)
			selected_columns_names4 = st.selectbox("Select Pearsonr Correlation",all_columns_names4)
			if st.button("Generate Pearsonr Correlation"):
				df=pd.DataFrame(Show_pearsonr(imp_df[selected_columns_name3],imp_df[selected_columns_names4]),index=['Pvalue', '0'])
				st.dataframe(df)  

			spearmanr3 = show_columns(df)
			spearmanr4 = show_columns(df)            
			spearmanr13 = st.selectbox("Select Columns spearmanr3",spearmanr4)
			spearmanr14 = st.selectbox("Select spearmanr4 Correlation",spearmanr4)
			if st.button("Generate spearmanr Correlation"):
				df=pd.DataFrame(Show_spearmanr(catego_df[spearmanr13],catego_df[spearmanr14]),index=['Pvalue', '0'])
				st.dataframe(df)

			st.subheader("UNIVARIATE ANALYSIS")
			
			all_columns_names = show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns ",all_columns_names)
			if st.checkbox("Show Histogram for variable"):
				st.write(show_hist(df[selected_columns_names]))
				st.pyplot()		

			all_columns_names = show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns Distplot ",all_columns_names)
			if st.checkbox("Show DisPlot for variable"):
				st.write(Show_DisPlot(df[selected_columns_names]))
				st.pyplot()

			all_columns_names = show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns CountPlot ",all_columns_names)
			if st.checkbox("Show CountPlot for variable"):
				st.write(Show_CountPlot(df[selected_columns_names]))
				st.pyplot()

			st.subheader("BIVARIATE ANALYSIS")

			Scatter1 = show_columns(df)
			Scatter2 = show_columns(df)            
			Scatter11 = st.selectbox("Select First Varibale",Scatter1)
			Scatter22 = st.selectbox("Select Second Varibale",Scatter2)
			if st.button("Generate PLOTLY Scatter PLOT"):
				st.pyplot(plotly(df,df[Scatter11],df[Scatter22]))
                
			bar1 = show_columns(df)
			bar2 = show_columns(df)            
			bar11 = st.selectbox("Select Varibale1",bar1)
			bar22 = st.selectbox("Select Varibale2",bar2)
			if st.button("Generate PLOTLY histogram PLOT"):
				st.pyplot(plotly_histogram(df,df[bar11],df[bar22]))                

			violin1 = show_columns(df)
			violin2 = show_columns(df)            
			violin11 = st.selectbox("violin1",violin1)
			violin22 = st.selectbox("violin2",violin2)
			if st.button("Generate PLOTLY violin PLOT"):
				st.pyplot(plotly_violin(df,df[violin11],df[violin22]))  

			st.subheader("MULTIVARIATE ANALYSIS")

			if st.checkbox("Show Histogram"):
				st.write(show_hist(df))
				st.pyplot()

			if st.checkbox("Show HeatMap"):
				st.write(Show_HeatMap(df))
				st.pyplot()

			if st.checkbox("Show PairPlot"):
				st.write(Show_PairPlot(df))
				st.pyplot()

			if st.button("Generate Word Cloud"):
				st.write(wordcloud(df))
				st.pyplot()

	elif choice == 'EDA For Linear Models':
		st.subheader("EDA For Linear Models")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx", "tsv"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")


			all_columns_names = show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns qqplot ",all_columns_names)
			if st.checkbox("Show qqplot for variable"):
				st.write(qqplot(df[selected_columns_names]))
				st.pyplot()

			all_columns_names = show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns outlier ",all_columns_names)
			if st.checkbox("Show outliers in variable"):
				st.write(outlier(df[selected_columns_names]))

			# all_columns_names = show_columns(df)         
			# selected_columns_names = st.selectbox("Select target ",all_columns_names)
			# if st.checkbox("Anderson Normality Test"):
			# 	st.write(Anderson_test(df[selected_columns_names]))	

			if st.checkbox("Show Distplot Selected Columns"):
				selected_columns_names = st.selectbox("Select Columns for Distplot ",all_columns_names)
				st.dataframe(show_displot(df[selected_columns_names]))
				st.pyplot()

			con1 = show_columns(df)
			con2 = show_columns(df)            
			conn1 = st.selectbox("Select 1st Columns for chi square test",con1)
			conn2 = st.selectbox("Select 2st Columns for chi square test",con2)
			if st.button("Generate chi square test"):
				st.write(check_cat_relation(df[conn1],df[conn2],0.5))
			


	elif choice == 'Feature Engineering':
		st.subheader("Feature Engineering")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")

			if st.checkbox("Show Selected Columns1"):
				selected_columns = st.multiselect("Select Columns1",show_columns(df))
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Numerical Variables"):
				num_df = Numerical_variables(df)
				numer_df=pd.DataFrame(num_df)                
				st.dataframe(numer_df)

			if st.checkbox("Categorical Variables"):
				new_df = categorical_variables(df)
				catego_df=pd.DataFrame(new_df)                
				st.dataframe(catego_df)

			if st.checkbox("DropNA Variable"):
				df = imputee(df)
				st.dataframe(df)

			if st.checkbox("Columns Missing after DropNA"):
				st.write(Show_Missing2(df))

			all_columns_names7 = show_columns(df)          
			selected_columns_names6 = st.selectbox("Select Columns FE ",all_columns_names7)
			if st.button("Label Encode"):
				label_df1=pd.DataFrame(label(df[selected_columns_names6]))
				st.dataframe(label_df1)
			

			all_columns_names7 = show_columns(df)           
			selected_columns_names5 = st.selectbox("Select Columns FE2 ",all_columns_names7)
			if st.button("Label Encode Var 2"):
				label_df2=pd.DataFrame(label(df[selected_columns_names5]))
				st.dataframe(label_df2)

			all_columns_names8 = show_columns(df)           
			selected_columns_names8 = st.selectbox("Select Columns FE3 ",all_columns_names8)
			if st.button("Label Encode Var 3"):
				label_df3=pd.DataFrame(label(df[selected_columns_names8]))
				st.dataframe(label_df3)

			if st.checkbox("Dummay Variable"):
				df = dummy(df)
				st.dataframe(df)

			if st.checkbox("Principle Component Analysis"):
				df = PCA(df,10)
				st.dataframe(df)

	elif choice == 'Model Building':
		st.subheader("Model Building [IN Processs]")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")

			if st.checkbox("Show Selected Columns for target"):
				selected_columns_ = st.multiselect("Select Columns for seperation",show_columns(df))
				sep_df = df[selected_columns_]
				st.dataframe(sep_df)

			if st.checkbox("Select Indpendent Data"):
				x = sep_df.iloc[:,:-1]
				st.dataframe(x)

			if st.checkbox("Select Dependent Data"):
				y = sep_df.iloc[:,-1]
				st.dataframe(y)
			if st.checkbox("Select X Train"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(x_train)

			if st.checkbox("Select x_test"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(x_test)

			if st.checkbox("Select y_train"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(y_train)

			if st.checkbox("Select y_test"):
				from sklearn.model_selection import train_test_split
				x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
				st.dataframe(y_test)

	st.markdown('Automation is **_really_ _cool_**.')
	st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
	st.title("Credits and Inspiration")
	"""https://pycaret.org/"""    


if __name__ == '__main__':
	main()
