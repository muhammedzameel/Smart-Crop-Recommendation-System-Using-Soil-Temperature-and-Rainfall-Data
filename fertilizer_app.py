
# Import required libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# ✅ Generate synthetic data for training (simulates agricultural features)
X_syn, y_syn = make_classification(
    n_samples=8000,          # number of samples
    n_features=8,            # total features
    n_informative=6,         # informative features for classification
    n_redundant=0,           # no redundant features
    n_classes=7,             # number of fertilizer classes
    weights=[1/7]*7,         # equal class distribution
    class_sep=2.0,           # separation between classes
    random_state=42
)

# 🔧 Create a DataFrame and add categorical columns
df_syn = pd.DataFrame(X_syn, columns=[
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Phosphorous', 'Potassium',
    'Feature1', 'Feature2'
])
df_syn['Soil Type'] = np.random.choice(['Sandy', 'Loamy', 'Black', 'Red'], size=8000)
df_syn['Crop Type'] = np.random.choice(['Maize', 'Wheat', 'Cotton', 'Sugarcane'], size=8000)

# 🎯 Define class labels for fertilizers
fertilizer_labels = ['Urea', 'DAP', '10-26-26', '17-17-17', '28-28', '20-20', '14-35-14']
df_syn['Fertilizer Name'] = LabelEncoder().fit(fertilizer_labels).inverse_transform(y_syn)

# 🧪 Prepare features and target
X = df_syn.drop(columns=['Fertilizer Name'])
y = df_syn['Fertilizer Name']

# 🔢 Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🔧 One-hot encode categorical input columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# 🧠 Create a pipeline: preprocessing + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        random_state=42
    ))
])

# 📊 Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
accuracy = accuracy_score(y_test, pipeline.predict(X_test))
print(f"✅ Model trained. Accuracy: {accuracy:.2f}")

# 💾 Save model and label encoder
with open('synthetic_fertilizer_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
with open('synthetic_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# 🔄 Load model and encoder for Streamlit app
with open('synthetic_fertilizer_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('synthetic_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 🌐 Streamlit app setup
st.set_page_config(page_title="🌾 Fertilizer Recommender", layout="centered")
st.title("🌱 Fertilizer Recommendation System")

# 📈 Sidebar: show model info
st.sidebar.header("🔍 Model Info")
st.sidebar.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

# 📋 Input section
st.subheader("📥 Enter Environmental and Crop Data")

# 🌡️ Collect user input
temperature = st.slider("Temperature (°C)", 10.0, 50.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0, 50.0)
moisture = st.slider("Moisture (%)", 0.0, 100.0, 40.0)
nitrogen = st.number_input("Nitrogen Level", 0, 100, 25)
phosphorous = st.number_input("Phosphorous Level", 0, 100, 25)
potassium = st.number_input("Potassium Level", 0, 100, 25)
soil_type = st.selectbox("Soil Type", ['Sandy', 'Loamy', 'Black', 'Red'])
crop_type = st.selectbox("Crop Type", ['Maize', 'Wheat', 'Cotton', 'Sugarcane'])
feature1 = st.slider("Feature1 (synthetic)", -3.0, 3.0, 0.0)
feature2 = st.slider("Feature2 (synthetic)", -3.0, 3.0, 0.0)

# 🚀 Predict button
if st.button("Predict Fertilizer"):
    # 🔍 Construct input DataFrame
    input_data = pd.DataFrame([{
        'Temparature': temperature,
        'Humidity': humidity,
        'Moisture': moisture,
        'Nitrogen': nitrogen,
        'Phosphorous': phosphorous,
        'Potassium': potassium,
        'Feature1': feature1,
        'Feature2': feature2,
        'Soil Type': soil_type,
        'Crop Type': crop_type
    }])

    # 🔮 Predict fertilizer
    prediction = model.predict(input_data)
    fertilizer_name = label_encoder.inverse_transform(prediction)[0]

    # 🎉 Display result
    st.balloons()
    st.success(f"🌾 Recommended Fertilizer: **{fertilizer_name}**")

    # 📑 Show input summary
    st.subheader("📋 Your Input Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}))

    # 🔢 Show top 3 predictions with confidence
    probabilities = model.predict_proba(input_data)[0]
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top3_fertilizers = label_encoder.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]

    st.subheader("📊 Top 3 Prediction Confidence")
    for fert, prob in zip(top3_fertilizers, top3_probs):
        st.write(f"**{fert}**: {prob * 100:.2f}%")
        st.progress(min(int(prob * 100), 100))

    # 📉 Plot top 5 predictions
    prob_df = pd.DataFrame({
        'Fertilizer': label_encoder.inverse_transform(np.arange(len(probabilities))),
        'Probability': probabilities
    }).sort_values(by='Probability', ascending=False).head(5)

    fig = px.bar(prob_df, x='Fertilizer', y='Probability', text='Probability',
                 labels={'Probability': 'Confidence Level'},
                 title='Top 5 Fertilizer Prediction Probabilities')
    st.plotly_chart(fig)
# Smart-Crop-Recommendation-System-Using-Soil-Temperature-and-Rainfall-Data
