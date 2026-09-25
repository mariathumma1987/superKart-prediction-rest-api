import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend inside the Docker network
BACKEND_URL = "https://curly-robot-xrppvq5jj5q6hp9wg-5000.app.github.dev"

# Set page configuration and title
st.set_page_config(page_title="SuperKart Sales Predictor", page_icon="🛒", layout="wide")
st.title("SuperKart Product Store Sales Prediction")
st.write("Predict total revenue and store sales for SuperKart retail items using machine learning.")

# Sidebar navigation
app_mode = st.sidebar.selectbox("Choose Prediction Mode", ["Online Prediction", "Batch Prediction"])

# ==========================================
# SECTION 1: ONLINE PREDICTION
# ==========================================
if app_mode == "Online Prediction":
    st.subheader("📊 Single Product Sales Prediction")
    st.markdown("Enter product and store characteristics below to estimate total sales revenue.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Product Attributes")
        product_weight = st.number_input("Product Weight (kg)", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
        product_sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Allocated Display Area Ratio", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        product_type = st.selectbox("Product Category", [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene",
            "Hard Drinks", "Canned Goods", "Breads", "Starchy Foods", "Others", "Seafood"
        ])
        product_mrp = st.number_input("Product MRP ($)", min_value=0.0, max_value=1000.0, value=140.0, step=1.0)
        product_category_code = st.selectbox("Product Category Code", ["FD", "DR", "NC"])

    with col2:
        st.markdown("### Store Attributes")
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
        store_age = st.number_input("Store Operating Age (Years)", min_value=0, max_value=100, value=27, step=1)

    # Convert inputs into payload dictionary matching Flask /v1/predict
    payload = {
        'Product_Weight': product_weight,
        'Product_Sugar_Content': product_sugar_content,
        'Product_Allocated_Area': product_allocated_area,
        'Product_Type': product_type,
        'Product_MRP': product_mrp,
        'Store_Size': store_size,
        'Store_Location_City_Type': store_location_city_type,
        'Store_Type': store_type,
        'Store_Age': int(store_age),
        'Product_Category_Code': product_category_code
    }

    st.markdown("---")
    if st.button("Predict Sales Revenue", type="primary"):
        try:
            with st.spinner("Calculating sales forecast..."):
                response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)

            if response.status_code == 200:
                result = response.json()
                predicted_sales = result['Predicted Store Sales Total ($)']
                st.success(f"### Predicted Total Store Sales: **${predicted_sales:,.2f}**")
            else:
                st.error(f"Error from API server: {response.text}")
        except Exception as e:
            st.error(f"Unable to connect to Flask API backend: {e}")

# ==========================================
# SECTION 2: BATCH PREDICTION
# ==========================================
else:
    st.subheader("Batch Predictions via CSV")
    st.markdown("Upload a CSV file containing multiple product listings to calculate total sales predictions in bulk.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Display preview of uploaded dataset
        df_preview = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview", df_preview.head(5))

        # Apply column renaming to match backend expectations
        df_preview = df_preview.rename(columns={
           "Product_Type_Category": "Product_Category_Code",
           "Product_Id_char":  "Product_Type",
           "Store_Age_Years" : "Store_Age"
        })

        # Reset file pointer after preview
        uploaded_file.seek(0)

        if st.button("Generate Batch Predictions", type="primary"):
            try:
                with st.spinner("Processing batch predictions..."):
                    # Convert the modified DataFrame back to CSV bytes for sending
                    csv_bytes = df_preview.to_csv(index=False).encode('utf-8')
                    files = {"file": ("batch_data.csv", csv_bytes, "text/csv")}
                    response = requests.post(f"{BACKEND_URL}/v1/predict_batch", files=files)

                if response.status_code == 200:
                    res_json = response.json()
                    st.success("Batch prediction completed successfully!")

                    predictions = res_json['predictions']

                    if isinstance(predictions, dict):
                        # Convert output dictionary into a structured table
                        pred_df = pd.DataFrame(list(predictions.items()), columns=['Product_Store_ID', 'Predicted_Sales_Total'])
                        st.dataframe(pred_df, use_container_width=True)

                        # Download button for results
                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Prediction Results CSV",
                            data=csv,
                            file_name="superkart_sales_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write(predictions)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Unable to connect to backend: {e}")
