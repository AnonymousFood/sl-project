from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df): # clean the data!

    # Create copy to avoid modifying original data
    df = df.copy()

    df['TotalSpent'] = df['RoomService'] + df['FoodCourt'] + \
                       df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['HasSpent'] = (df['TotalSpent'] > 0).astype(int)
    
    # Convert boolean values to strings for categorical encoding
    bool_cols = ['CryoSleep', 'VIP']
    for col in bool_cols:
        df[col] = df[col].fillna('Unknown')
        df[col] = df[col].astype(str)
    
    # Handle categorical variables
    le = LabelEncoder()
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'HasSpent']
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = le.fit_transform(df[col])
    
    # Extract deck, num, side from Cabin
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Num'] = df['Cabin'].str.split('/').str[1].fillna(-1).astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]
    
    # Encode new categorical variables
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')
    df['Deck'] = le.fit_transform(df['Deck'])
    df['Side'] = le.fit_transform(df['Side'])
    
    # Handle numerical variables
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'TotalSpent']
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Select features for model
    features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
               'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
               'Deck', 'Num', 'Side', 'HasSpent', 'TotalSpent']
    
    return df[features]