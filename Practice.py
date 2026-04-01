import pandas as pd
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt 

logging.basicConfig(
    filename="Cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def Automate_Cleaning(Path):
    df=pd.read_csv(Path)

    logging.info("Cleaning Proccess Started")

    Before=len(df)
    df.drop_duplicates(inplace=True)
    After=len(df)
    logging.info(f"Drop {Before-After} Duplicate Rows")
    df.columns=df.columns.str.strip().str.lower().str.replace(" ","_")
    for col in df.columns:
        df['Age'] = df['age'].apply(lambda x: x if x <= 90 else np.nan)
        df["Age"].fillna(df['Age'].median(),inplace=True)
        if np.issubdtype(df[col].dtype,np.number):
            negative=(df[col] < 0).sum()
            if negative > 0:
                df.loc[df[col] < 0, col] = np.nan
                logging.info(f"Detected {negative} negative values in {col}")
            if df[col].max() <= 10 and df[col].min() >= 0:
                df[col] = df[col].clip(0,5)

            missing=df[col].isnull().sum()
            Q1=df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)
            IQR=Q3-Q1
            Lower_bound=Q1-3*IQR
            Upper_Bound=Q3+3*IQR
            Outliers=((df[col]<Lower_bound)|(df[col]>Upper_Bound)).sum()
            logging.info(f"Detect {Outliers} outliers in column {col}")

            df.loc[(df[col]<Lower_bound)|(df[col]>Upper_Bound),col]=df[col].median()
            logging.info(f"Handled {Outliers} outliers in column {col}")
            df[col].fillna(df[col].mean(),inplace=True)
            logging.info(f"Clean {missing} Missing values from {col}")
            df[col]=df[col].round(2)
        elif np.issubdtype(df[col].dtype,np.object_):
            df[col]=df[col].str.replace("_","")
            df[col]=df[col].str.strip()
            if "date" in df[col]:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            missing_object=df[col].isnull().sum()
            df[col]=df[col].str.strip().str.lower()
            df.replace(["NaN","unknown","not_a_date","mle","fmale","losangeles"],np.nan,inplace=True)
            df[col].fillna(df[col].mode()[0],inplace=True)
            logging.info(f"Clean {missing_object} missing values From column {col}")
    return df
clean=Automate_Cleaning("Sales.csv") 
clean.to_csv("clean.csv",index=False)
column=clean.columns.to_list()
numeric_column=clean.select_dtypes(include=[np.number]).columns[:6]
string_column=clean.select_dtypes(include=[np.object_]).columns[:6]
print(string_column)
print(numeric_column)
fig,ax=plt.subplots(2,2,figsize=(20,25))
plt.suptitle('Sales Data Distribution Analysis', fontsize=16, fontweight='bold')
sns.pointplot(data=clean,x=string_column[5],y=numeric_column[1],ax=ax[0,0],color='crimson')
ax[0,0].set_title(f"Distribution of {string_column[5]}")
sns.boxplot(data=clean,y=numeric_column[0], ax=ax[0, 1], color='skyblue',
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8, 'markeredgecolor': 'darkred', 'alpha': 0.7})
ax[0,1].set_title(f"Outliers in column {numeric_column[0]} Before Cleaning")
col_name=string_column[3]
counts=clean[col_name].value_counts()
ax[1,0].pie(counts, labels=counts.index, autopct='%1.1f%%',startangle=120)
ax[1,0].legend(title=col_name,  loc="center left", bbox_to_anchor=(-0.5, 0.5))
sns.boxplot(data=clean,y=numeric_column[3], ax=ax[1, 1], color='skyblue',
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8, 'markeredgecolor': 'darkred', 'alpha': 0.7})
ax[1,1].set_title(f"Outliers in column {numeric_column[0]} After cleaning Cleaning")
plt.show()
plt.savefig("Sales_Analysis.png")