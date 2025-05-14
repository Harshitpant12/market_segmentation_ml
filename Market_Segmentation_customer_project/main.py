import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the trained clustering model
filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Load the customer dataset
df = pd.read_csv("Clustered_Customer_Data.csv", index_col=0)


# Set up the Streamlit app
st.markdown(
    '<style>body { background-color: #e6f2ff; }</style>',
    unsafe_allow_html=True
)
st.title("📊 Market Segmentation using Clustering")

# Input form
with st.form("customer_input_form"):
    st.subheader("Enter Customer Data")
    balance = st.number_input('Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input('Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input('Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input('OneOff Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input('Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input('Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input('Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input('Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input('Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input('Cash Advance Transactions', step=1)
    purchases_trx = st.number_input('Purchases Transactions', step=1)
    credit_limit = st.number_input('Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input('Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input('Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input('PRC Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input('Tenure', step=1)

    submitted = st.form_submit_button("Submit")

# If form is submitted
if submitted:
    # Prepare input data for prediction
    input_data = [[
        balance, balance_frequency, purchases, oneoff_purchases,
        installments_purchases, cash_advance, purchases_frequency,
        oneoff_purchases_frequency, purchases_installment_frequency,
        cash_advance_frequency, cash_advance_trx, purchases_trx,
        credit_limit, payments, minimum_payments, prc_full_payment, tenure
    ]]

    # Predict cluster
    predicted_cluster = loaded_model.predict(input_data)[0]
    st.success(f"✅ This customer belongs to **Cluster {predicted_cluster}**")

    # Filter data for that cluster
    cluster_df = df[df['Cluster'] == predicted_cluster]

    # Plot histograms of all features in this cluster
    st.subheader(f"📈 Feature Distributions in Cluster {predicted_cluster}")
    plt.rcParams["figure.figsize"] = (6, 4)

    for col in cluster_df.drop('Cluster', axis=1).columns:
        fig, ax = plt.subplots()
        sns.histplot(cluster_df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
