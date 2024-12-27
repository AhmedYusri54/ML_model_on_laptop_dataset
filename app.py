import pandas as pd
import streamlit as st
import pickle as pk
import seaborn as sns
from helper_funcation import  dis_char, count_features_plot
import matplotlib.pyplot as plt
import sklearn
# Load the dataset
df = pd.read_csv("final_laptop_price.csv")

# Drop the Unnamed column from the dataset 
df.drop(columns="Unnamed: 0", inplace=True)
st.title(f"Machine Learning Regression model on Laptops dataset.")
website_link = "https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset"
st.write("The Dataset link: [Laptops dataset](%s)" % website_link)
st.sidebar.header("Navigation")
st.sidebar.markdown("Created by [Ahmed Yusri](https://www.linkedin.com/in/ahmed-yusri-499a67313)")

# Create an set of options for user to select
sidebar_option = st.sidebar.radio("Choose an Option:", ["Overview", "EDA", "Modeling", "Insights"])

if sidebar_option == "Overview":
    st.header("Data Overview")
    st.write("The features Names and it's data types.")
    st.write(df.describe(include='all'))
    for col in df.columns:
        st.write(f"{col}: ({df[col].dtype})")
        
elif sidebar_option == "EDA":
    st.header("Exploratory Data Analysis")
    
    # 1. Put a select radio to choose to see a uni or bivariant analysis
    analysis_type_option = st.sidebar.radio("Choose type of Analysis:", ["Feature distribution analysis", "Relation between dependent and independent"])
    if analysis_type_option == "Feature distribution analysis":
        st.subheader("Interactive Feature Distribution in the dataset")
        column_names = list(df.columns)
        selected_col = st.sidebar.selectbox(f"Select column for analysis", column_names)
        st.pyplot(count_features_plot(data=df, feature_name=selected_col))              
        st.write("As i see each plot gives the distribution of the values of features in the dataset.")
        
    elif analysis_type_option == "Relation between dependent and independent":
        st.subheader("Interactive Visualizations to see Relation between dependent and independent features")
        column_names = list(df.columns)
        def_idx = column_names.index("Price_EGP")
        selected_feature = st.sidebar.selectbox(f"Select Frist Feature", column_names)
        new_names = column_names.remove(selected_feature) 
        selected_target = st.sidebar.selectbox(f"Select the Target", column_names)
        st.pyplot(dis_char(df, selected_feature, selected_target))
        st.write("So as i see the features most of them as a direct relation between them and price as target")
        st.markdown("### Let's see the Heatmap plot to get best look in the relationships")
        corr_matrix = df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.write("As i see that the correlation between the features and price is low unless the RAM is 0.74 so it is relatvily high.")
        st.write("The correlation between `inch` and `weight` is 0.83 so it maybe `Multicollinearity`.")
        
elif sidebar_option == "Modeling":
    st.subheader("After Analyizing the data and modelling the best model is `CatBoostRegression`")
    st.write("Les't Predict a Price of a laptop.")
    select_company =  st.selectbox("Choose a company", df["Company"].unique())
    type_input = st.text_input("Write the product name of your laptop", "etc. Legion 5")
    select_type = st.selectbox("Choose the type of the laptop", df["TypeName"].unique())
    inches_input = st.number_input("Type the inches")
    res_input = st.text_input("Write the Screen Resolution you want", "etc. 1440x1920")
    select_cpu_comp = st.selectbox("Choose the CPU Company", df["CPU_Company"].unique())
    cpu_input = st.text_input("Write the CPU type", "etc, Corei5")
    feq_input = st.number_input("Write the CPU Frequency in GHz")
    select_ram = int(st.selectbox("Choose the RAM Capacity", df["RAM (GB)"].unique()))
    memory_input = st.text_input("Write the Memory Capacity", "in GB and SSD or HDD")
    select_gpu_comp = st.selectbox("Choose the GPU Company", df["GPU_Company"].unique())
    gpu_input = st.text_input("Write the GPU type")
    select_os = st.selectbox("Choose your Operating System", df["OpSys"].unique())
    weights_input = st.number_input("Write the weight of the laptop")
    pred_list = [45, 85.5, 95 ,108, 76, 25, 23.2]
    user_choose = {
        "Company": select_company,
        "Product": type_input,
        "TypeName": select_type,
        "Inches": inches_input,
        "ScreenResolution": res_input,
        "CPU_Company": select_cpu_comp,
        "CPU_Type": cpu_input,
        "CPU_Frequency (GHz)": feq_input,
        "RAM (GB)": select_ram,
        "Memory": memory_input,
        "GPU_Company": select_gpu_comp,
        "GPU_Type": gpu_input,
        "OpSys": select_os,
        "Weight (kg)": weights_input,
    }
    user_df = pd.DataFrame(user_choose, index=[0])
    st.write(user_df)
    # Load the model 
    with open("LR_model.pkl", "rb") as f:
        model = pk.load(f)
    st.write(user_df.dtypes)
    
    # Load the encoders 
    with open("encoders.pkl", "rb") as f:
        encoders = pk.load(f)
    # transform the data using encode function
  
    num_list = ["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)"]
    # Encode categorical features
    for col in user_df.columns:
            if col not in num_list:
                if col in encoders:
                    encoder = encoders[col]
                    if user_df[col].iloc[0] in encoder.classes_:
                            user_df[col] = encoder.transform(user_df[col])
                    else:
                            user_df[col] = -1   
                            
    st.write(user_df)   
    with open("scaler.pkl", "rb") as f:
        scaler = pk.load(f)
        st.write("Scaled data")
    user_df = user_df[df.drop(columns="Price_EGP").columns]    
            
    user_scaler = scaler.transform(user_df)
            
    st.write(user_scaler)
         
    if st.button("Predict"):
        y_pred = model.predict(user_scaler)
        st.write(f"The Price of your laptop is: {round(y_pred[0][0] * 100 + 70, 2)}k EGP")   
elif sidebar_option == "Insights":
    st.subheader("My Insights after training and Deployment the model.")
    st.write("The Dataset is all most have a categorical data so it must apply so encoding and Scalling for better performance.")
    st.write("The Random forest model Gives me go accuracy but it doesn't mean it is good.")
    st.write("Comparing models results gives a good understanding of the best model to choose.")       
