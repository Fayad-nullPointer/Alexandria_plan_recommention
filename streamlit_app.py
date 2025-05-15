import streamlit as st
import pandas as pd
import requests
import io
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os
import random

# Set page configuration
st.set_page_config(page_title="Alexandria Tour Guide Recommender", layout="wide")

# Custom CSS to improve the appearance
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.recommendation-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
.place-name {
    color: #1e3d59;
    font-size: 24px;
    font-weight: bold;
}
.hotel-name {
    color: #ff6e40;
    font-weight: bold;
}
.image-container {
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('tour_guide_plan_alexandria (1).csv')
    return df

# Function to preprocess data for the recommender system
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Handle tourist schedules (convert to a list of destinations)
    df_processed['Tourist Schedule List'] = df_processed['Tourist Schedule of Visits'].str.split(', ')
    
    # One-hot encode categorical features
    df_processed['Historical'] = df_processed['Historical or Not'].apply(lambda x: 1 if x == 'Yes' else 0)
    df_processed['Cinema'] = df_processed['Cinema or Not'].apply(lambda x: 1 if x == 'Yes' else 0)
    df_processed['Sea'] = df_processed['Place on Sea'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Extract hotel location from hotel address
    df_processed['Hotel Location'] = df_processed['Hotel address'].str.extract(r'(\w+ St|Beach)', expand=False)
    df_processed['Hotel Location'].fillna('Other', inplace=True)
    
    # One-hot encode hotel locations
    hotel_encoded = pd.get_dummies(df_processed['Hotel Location'], prefix='Hotel')
    
    # Convert time preferences to numerical (hours)
    df_processed['Time Preferred Hours'] = df_processed['Time Preferred to Go Out'].apply(
        lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60
    )
    
    # Normalize time preference (0 to 1)
    min_time = df_processed['Time Preferred Hours'].min()
    max_time = df_processed['Time Preferred Hours'].max()
    df_processed['Time Preferred Normalized'] = (df_processed['Time Preferred Hours'] - min_time) / (max_time - min_time)
    
    # One-hot encode tourist destinations
    mlb = MultiLabelBinarizer()
    tourist_encoded = pd.DataFrame(
        mlb.fit_transform(df_processed['Tourist Schedule List']),
        columns=mlb.classes_,
        index=df_processed.index
    )
    
    # Combine all features for the recommender system
    features = pd.concat([
        df_processed[['Historical', 'Cinema', 'Sea', 'Time Preferred Normalized']],
        df_processed['Family Members Number'].apply(lambda x: min(x / 6, 1)),  # Normalize family size (assuming max of 6)
        hotel_encoded,  # Add hotel location features
        tourist_encoded
    ], axis=1)
    
    return df_processed, features, min_time, max_time, mlb.classes_

# Function to get recommendations based on user preferences
def get_recommendations_by_preferences(
    features,
    df,
    min_time,
    max_time,
    historical_preference='Yes',
    cinema_preference='No',
    sea_preference='Yes',
    time_preference='11:00',
    family_size=2,
    preferred_destinations=None,
    preferred_hotel=None,
    n=5
):
    """Get tour plan recommendations based on user preferences"""
    # Create a user preference vector
    user_pref = {}
    
    # Process categorical preferences
    user_pref['Historical'] = 1 if historical_preference == 'Yes' else 0
    user_pref['Cinema'] = 1 if cinema_preference == 'Yes' else 0
    user_pref['Sea'] = 1 if sea_preference == 'Yes' else 0
    
    # Process time preference
    time_hours = float(time_preference.split(':')[0]) + float(time_preference.split(':')[1])/60
    user_pref['Time Preferred Normalized'] = (time_hours - min_time) / (max_time - min_time)
    
    # Process family size
    user_pref['Family Members Number'] = min(family_size / 6, 1)  # Normalize
    
    # Create user preference DataFrame with the same columns as features
    user_pref_df = pd.DataFrame({col: 0 for col in features.columns}, index=[0])
    
    # Update with user preferences
    for col, val in user_pref.items():
        if col in user_pref_df.columns:
            user_pref_df[col] = val
    
    # Add preferred destinations
    if preferred_destinations:
        for dest in preferred_destinations:
            if dest in user_pref_df.columns:
                user_pref_df[dest] = 1
    
    # Add preferred hotel location
    if preferred_hotel:
        # Extract location from hotel address
        hotel_location = None
        if 'Corniche' in preferred_hotel:
            hotel_location = 'Corniche St'
        elif 'Sidi Gaber' in preferred_hotel:
            hotel_location = 'Sidi Gaber St'
        elif 'El Horreya' in preferred_hotel:
            hotel_location = 'El Horreya St'
        elif 'Al Agamy' in preferred_hotel:
            hotel_location = 'Al Agamy St'
        elif 'Stanley' in preferred_hotel:
            hotel_location = 'Stanley St'
        
        # Set hotel location preference
        if hotel_location:
            hotel_col = f'Hotel_{hotel_location}'
            if hotel_col in user_pref_df.columns:
                user_pref_df[hotel_col] = 1
    
    # Calculate similarity with all tour plans
    user_similarity = cosine_similarity(user_pref_df, features)[0]
    
    # Get indices of most similar plans
    similar_indices = (-user_similarity).argsort()[:n]
    
    # Return the recommended plans
    return df.iloc[similar_indices], user_similarity[similar_indices]

# We'll use images directly from the image_cache folder instead of downloading them

# Function to get an image directly from the image_cache folder
def get_place_image(place_name):
    # Clean up place name for lookup
    clean_name = place_name.strip().replace(" Alexandria", "")
    
    # Create a safe filename like the one used when saving to cache
    safe_name = ''.join(c if c.isalnum() else '_' for c in clean_name)
    cache_path = f"image_cache/{safe_name}.jpg"
    
    # Try to load the image from the cache
    try:
        if os.path.exists(cache_path):
            return Image.open(cache_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
    
    # If not found with exact name, try partial matching with existing files
    try:
        cache_files = os.listdir("image_cache")
        for file in cache_files:
            file_name = file.replace(".jpg", "").replace("_", " ")
            if (clean_name.lower() in file_name.lower() or 
                file_name.lower() in clean_name.lower()):
                return Image.open(os.path.join("image_cache", file))
    except Exception as e:
        pass
    
    # Return None if all attempts fail
    return None

# Cache directory for images
os.makedirs('image_cache', exist_ok=True)

# Function to get a cached image from the image_cache directory
def get_cached_image(place_name):
    # Get just the place name without "Alexandria"
    clean_name = place_name.replace(" Alexandria", "").strip()
    
    # First try direct lookup using the cleaned place name
    direct_img = get_place_image(clean_name)
    if direct_img:
        return direct_img
    
    # If no direct match, try matching with known place names
    place_mapping = {
        "qaitbay citadel": "Qaitbay_Citadel.jpg",
        "montazah palace": "Montazah_Palace_Gardens.jpg",
        "roman amphitheatre": "Roman_Amphitheatre.jpg",
        "alexandria national museum": "Alexandria_National_Museum_Alexandria.jpg",
        "stanley beach": "Stanley_Beach.jpg",
        "mamoura beach": "Mamoura_Beach.jpg",
        "pompey's pillar": "Pompey_s_Pillar_Alexandria.jpg",
        "pompeys pillar": "Pompey_s_Pillar_Alexandria.jpg",
        "zahran cinema": "Zahran_Cinema.jpg"
    }
    
    # Try to find a match in our mapping
    clean_lower = clean_name.lower()
    for key, filename in place_mapping.items():
        if key in clean_lower or clean_lower in key:
            try:
                return Image.open(os.path.join("image_cache", filename))
            except:
                pass
    
    # If no matches found in mapping, try looking through all files in the directory
    try:
        cache_files = os.listdir("image_cache")
        for file in cache_files:
            file_name = file.replace(".jpg", "").replace("_", " ").lower()
            if clean_lower in file_name or any(word in file_name for word in clean_lower.split()):
                return Image.open(os.path.join("image_cache", file))
    except:
        pass
    
    return None

# Ensure image cache directory exists
if not os.path.exists("image_cache"):
    os.makedirs("image_cache", exist_ok=True)
    
# Main application
def main():
    # Title and description
    st.title("Alexandria Tour Guide Plan Recommender")
    st.markdown("""
    This app helps you find the perfect Alexandria tour plan based on your preferences.
    Fill in your preferences below and get personalized recommendations!
    """)
    
    # Load and preprocess data
    df = load_data()
    df_processed, features, min_time, max_time, all_destinations = preprocess_data(df)
    
    # All available hotels
    all_hotels = [
        "Corniche St, Alexandria, Egypt",
        "Sidi Gaber St, Alexandria, Egypt",
        "El Horreya St, Alexandria, Egypt",
        "Al Agamy St, Alexandria, Egypt",
        "Stanley St, Alexandria, Egypt"
    ]
    
    # Sidebar for user preferences
    st.sidebar.header("Your Preferences")
    
    # Get user preferences
    historical = st.sidebar.selectbox("Do you prefer historical places?", ["Yes", "No"], index=0)
    cinema = st.sidebar.selectbox("Include cinema in your tour?", ["No", "Yes"], index=0)
    sea = st.sidebar.selectbox("Prefer places by the sea?", ["Yes", "No"], index=0)
    time_pref = st.sidebar.time_input("What time do you prefer to go out?", value=pd.to_datetime("11:00"))
    time_pref_str = f"{time_pref.hour:02d}:{time_pref.minute:02d}"
    family_size = st.sidebar.slider("How many family members will join?", 1, 6, 2)
    
    # Hotel preference
    preferred_hotel = st.sidebar.selectbox("Select your preferred hotel location:", all_hotels)
    
    # Destination preferences
    selected_destinations = st.sidebar.multiselect("Select your preferred destinations:", all_destinations)
    
    # Number of recommendations
    n_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 3)
    
    # Generate recommendations button
    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            # Get recommendations
            recommended_plans, similarity_scores = get_recommendations_by_preferences(
                features=features,
                df=df,
                min_time=min_time,
                max_time=max_time,
                historical_preference=historical,
                cinema_preference=cinema,
                sea_preference=sea,
                time_preference=time_pref_str,
                family_size=family_size,
                preferred_destinations=selected_destinations,
                preferred_hotel=preferred_hotel,
                n=n_recs
            )
            
            # Display recommendations
            st.subheader("Your Personalized Tour Recommendations")
            
            for i, (idx, row) in enumerate(recommended_plans.iterrows()):
                # Start a card with a container
                with st.container():
                    st.markdown(f"<div class='recommendation-card'>", unsafe_allow_html=True)
                    
                    # Create two columns within the card
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"<h3 class='place-name'>Recommendation {i+1}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Similarity Score:</strong> {similarity_scores[i]:.4f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Date:</strong> {row['Date']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong class='hotel-name'>Hotel:</strong> {row['Hotel address']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Tourist Schedule:</strong> {row['Tourist Schedule of Visits']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Historical:</strong> {row['Historical or Not']}, <strong>Cinema:</strong> {row['Cinema or Not']}, <strong>Sea:</strong> {row['Place on Sea']}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p><strong>Time:</strong> {row['Time Preferred to Go Out']}, <strong>Family Size:</strong> {row['Family Members Number']}</p>", unsafe_allow_html=True)
                            
                        with col2:
                            # Get destinations from the schedule
                            destinations = row['Tourist Schedule of Visits'].split(', ')
                            
                            # Display image for first destination if available
                            if destinations:
                                first_dest = destinations[0]
                                st.subheader(f"Images of {first_dest}")
                                # Get the image directly from our cache
                                img = get_cached_image(first_dest)
                                if img:
                                    st.image(img, caption=first_dest, use_container_width=True)
                                else:
                                    st.error(f"Could not load image for {first_dest}")
                            
                            # Show additional destinations if there are more than one
                            if len(destinations) > 1:
                                second_dest = destinations[1]
                                st.subheader(f"Images of {second_dest}")
                                # Get the image directly from our cache
                                img = get_cached_image(second_dest)
                                if img:
                                    st.image(img, caption=second_dest, use_container_width=True)
                                else:
                                    st.error(f"Could not load image for {second_dest}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")
    else:
        st.info("👈 Set your preferences in the sidebar and click 'Get Recommendations' to see personalized tour plans")        # Show some Alexandria attractions preview
        st.subheader("Popular Alexandria Attractions")
        preview_places = ["Qaitbay Citadel", "Montazah Palace Gardens", "Roman Amphitheatre"]
        
        # Show all cached images we have available
        available_images = {}
        try:
            for file in os.listdir("image_cache"):
                if file.endswith(".jpg"):
                    clean_name = file.replace(".jpg", "").replace("_", " ")
                    available_images[clean_name] = os.path.join("image_cache", file)
        except:
            pass
        
        # Display preview images
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        # Use our defined preview places if possible
        for i, place in enumerate(preview_places):
            if i < len(cols):
                with cols[i]:
                    img = get_cached_image(place)
                    if img:
                        st.image(img, caption=place, use_container_width=True)

if __name__ == "__main__":
    main()
