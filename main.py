import pandas as pd 

def process_data():
    try: 
        data = pd.read_csv('migration.csv') 
        df = pd.DataFrame(data)
        print(df.head())
        df_clean = df.dropna()
        print(f'After cleansing, the row numbwer is: {len(df_clean)}')
    except Exception as p: 
        print(f'There is no any path: {p}')


if __name__=='__main__':
    process_data()

