import pandas as pd 
def load_clean_data(file):
  df=pd.read_csv(file,encoding="latin-1",sep="\t",header=None)
  copy_df=df[[0,1]].copy()
  copy_df.columns=["label","message"]
  return copy_df

if __name__=="__main__":
  data=load_clean_data("data/raw/spam.csv")
  print(data)
  print("SUCCESS: Data loaded!")
  print("First 5 rows:")
  print(data.head())
  print("Last 5 rows:")
  print(data.tail())
  print("random 3 rows :")
  print(data.sample(3)) 
    
  print(f" Dataset shape: {data.shape}")
  print(f" Class distribution:\n{data['label'].value_counts()}")
