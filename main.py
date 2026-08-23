import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
def data_processing():
    data = pd.read_csv('Housing.csv')
    # print(data.head().to_string())
    Q1 = data['price'].quantile(0.25)
    Q3 = data['price'].quantile(0.75)
    IQR = Q3 - Q1
    
    data = data[(data['price'] >= Q1 - 1.5 * IQR) & (data['price'] <= Q3 + 1.5 * IQR)]


    # print('\n---- Missing Values ----')
    # print(data.isnull().sum())
    yes_no_values = ['mainroad', 'guestroom', 'prefarea', 'basement' ,'hotwaterheating', 'airconditioning']
    for binary_encoding in yes_no_values: 
        data[binary_encoding] = data[binary_encoding].replace({'no': 0, 'yes': 1})

    print('\n---- Yes/No columns Converted to Binary ----')
    # print(data.head().to_string())
    # print(data['furnishingstatus'].to_string())

    final_data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True, dtype=int)
    print('\n---- New columns after encoding ----')
    print(final_data.head().to_string())

    scaler = MinMaxScaler()

    scalered_features = scaler.fit_transform(final_data)
    print(scalered_features)
    scaled_df = pd.DataFrame(scalered_features, columns=final_data.columns)
    print('\n---- Final Cleaned and Scaled Data Ready for AI! ----')
    print(scaled_df.head().to_string())

    X = scaled_df.drop(columns=['price'])
    Y = scaled_df['price']
    print('\n---- X (Features) Column names ----')

    print('\n---- Y (Target) Shape')
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)
    print("\n---- Splitting Complete ----")
    print("Training clues shape:", X_train.shape)
    print("Training clues shape:", Y_test.shape)

    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    # print('Predicted prices by AI:', predictions[:5])
    # print('Actual prices:', Y_test.head().tolist())
    # print(data.head().to_string())  

    price_min = data['price'].min()
    price_max = data['price'].max()

    euros = Y_test.head() * (price_max - price_min) + price_min
    p_euros = predictions[:5] * (price_max - price_min) + price_min

    print(data.head().to_string())



    
    print('\n==================================================')
    print('   🤖 Ai Predictions and Actual Prices   ')
    print('==================================================')
    for i in range(5):
        print(f"#{i+1}| AI Predictions: EUR {p_euros[i]:,.2f} | Actual Prices: EUR {euros.iloc[i]:,.2f}")

    accuracy = r2_score(Y_test, predictions)
    
    print('\n==================================================')
    print(f"📊 AI Models' overall accuracy of R² Score: {accuracy * 100:.2f}%")
    print('==================================================')

        # === MODEL AND SCALER SERIALIZATION ===
   
    
    # Save the trained AI model
    with open('HPP.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    # Save the scaler asset
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
        
    print("\n💾 Model and Scaler successfully saved! (.pkl files created)")









if __name__ == '__main__':
    data_processing()