import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import hashlib

# Set page configurations
st.set_page_config(
    page_title="Election Outcome Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# User credentials (replace with your own)
valid_username = "user123"
valid_password_hash = hashlib.sha256("password123".encode()).hexdigest()


voting_df = None

# Upload file through Streamlit
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    voting_df = pd.read_csv(uploaded_file)
    st.write(voting_df)

# Use the uploaded data or your default dataset
#voting_url = r'C:\Users\Silicon\Downloads\Election_Data\Tamil_Nadu_State_Elections_2021_Details.csv'

# voting_url = uploaded_file
column_names = [
    'Constituency','Candidate','Party','EVM_Votes','Postal_Votes','Total_Votes','%_of_Votes',
    'Tot_Constituency_votes_polled','Tot_votes_by_parties','Winning_votes','Win_Lost_Flag'
]
#voting_df = pd.read_csv(voting_url, header=None, names=column_names)
if voting_df is not None:
    voting_df = voting_df.dropna()
    df = voting_df.drop(columns=['Constituency', 'Candidate', 'Party', 'Winning_votes'])
else:
    st.warning("Please upload a file to proceed or use the default dataset.")
    st.stop()
# Drop rows with missing values
df = voting_df.dropna()

# Drop non-numeric columns (adjust as needed)
df = df.drop(columns=['Constituency', 'Candidate', 'Party', 'Winning_votes'])

# Split data into features (X) and target variable (y)
X = df.drop('Win_Lost_Flag', axis=1)
y = df['Win_Lost_Flag']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Login Page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == valid_username and hashlib.sha256(password.encode()).hexdigest() == valid_password_hash:
            st.success("Login Successful!")
            st.session_state.login = True
        else:
            st.error("Invalid Credentials. Please try again.")

# Prediction Page
def prediction_page():
    st.title("Prediction Page")
    evm_votes = st.number_input("EVM Votes", min_value=0)
    postal_votes = st.number_input("Postal Votes", min_value=0)
    total_votes = st.number_input("Total Votes", min_value=0)
    percent_of_votes = st.number_input("Percentage of Votes", min_value=0.0, max_value=1.0, step=0.01)
    constituency_votes_polled = st.number_input("Constituency Votes Polled", min_value=0)
    total_votes_by_parties = st.number_input("Total Votes by Parties", min_value=0)

    # evm_votes_hashed = hashlib.sha256(str(evm_votes).encode()).hexdigest()
    # postal_votes_hashed = hashlib.sha256(str(postal_votes).encode()).hexdigest()
    # total_votes_hashed = hashlib.sha256(str(total_votes).encode()).hexdigest()
    # percent_of_votes_hashed = hashlib.sha256(str(percent_of_votes).encode()).hexdigest()
    # constituency_votes_polled_hashed = hashlib.sha256(str(constituency_votes_polled).encode()).hexdigest()
    # total_votes_by_parties_hashed = hashlib.sha256(str(total_votes_by_parties).encode()).hexdigest()

    # Create a sample data point for prediction
    sample_data = pd.DataFrame({
        'EVM_Votes': [evm_votes],
        'Postal_Votes': [postal_votes],
        'Total_Votes': [total_votes],
        '%_of_Votes': [percent_of_votes],
        'Tot_Constituency_votes_polled': [constituency_votes_polled],
        'Tot_votes_by_parties': [total_votes_by_parties]
    })

   
    # Ensure that the feature names match the ones used during training
    missing_features = set(X.columns) - set(sample_data.columns)
    if missing_features:
        # Add missing columns with zeros
        zeros_df = pd.DataFrame(0, index=sample_data.index, columns=list(missing_features))
        sample_data = pd.concat([sample_data, zeros_df], axis=1)

    # Reorder columns to match the original order during training
    sample_data = sample_data[X.columns]

    # Predict function
    def predict():
        prediction = model.predict(sample_data)
        return prediction[0]

    # Predict button
    if st.button("Predict"):
        result = predict()
        st.success(f"Predicted Win/Loss Flag: {result}")

# Main app
def main():
    if "login" not in st.session_state:
        st.session_state.login = False

    if not st.session_state.login:
        login_page()
    else:
        prediction_page()

if __name__ == "__main__":
    main()

