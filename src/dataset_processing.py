import os
import yaml
import argparse
import random
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from get_data import get_data,read_params
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)

def get_emotion_scores(text):
    results = classifier(text)
    emotion_dict = {emotion['label']: round(emotion['score'] * 100, 2) for emotion in results[0]}
    return emotion_dict

model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight but effectiv

def calculate_similarity(text, trend1, trend2):
    embeddings = model.encode([text, trend1, trend2], convert_to_tensor=True)
    similarity1 = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    similarity2 = util.pytorch_cos_sim(embeddings[0], embeddings[2]).item()
    
    avg_similarity = (similarity1 + similarity2) / 2  # Take the average of both scores
    print(f"Similarity1: {round(similarity1 * 100, 2)}%, Similarity2: {round(similarity2 * 100, 2)}%")
    
    return round(avg_similarity * 100, 2)

def extract_entities(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(str(text))  
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities if entities else None  

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(text))
    return score["compound"]

def convert_age_range_to_avg(df, column_name):
    def process_value(x, index):
        try:
            return sum(map(int, x.split('-'))) / 2 if '-' in x else float(x)
        except ValueError:
            print(f"Error in row {index}: '{x}'")  # Print the row index and value
            return None  # You can return None, NaN, or a default value
    
    df['AgeRange'] = [process_value(val, idx) for idx, val in enumerate(df[column_name])]
    return df

    # Auto-update mappings
def get_or_create_mapping(value, mapping_dict):
    """Returns the mapped ID if exists, else creates a new one."""
    if value in mapping_dict:
        return mapping_dict[value]
    else:
        new_id = max(mapping_dict.values(), default=0) + 1  # Assign next ID
        mapping_dict[value] = new_id
        return new_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../params.yaml')
    args = parser.parse_args()
    config_file=args.config
    config = get_data(config_file)
    
    csv_file = os.path.join(config['data']['csv_file'])
    df = pd.read_csv(csv_file)
    
    # Apply NER and sentiment analysis
    df["NER_Entities"] = df['CampaignText'].apply(extract_entities)
    df["SentimentScore"] = df['CampaignText'].apply(get_sentiment)

    platform_mapping = {
        "Twitter": 1,
        "Instagram": 2,
        "Linkedin": 3,
        "LinkedIn": 3,
        "Meta": 4,
        "Reddit": 5,
        "YouTube": 6,
        "Facebook":7,
        "TikTok":8
    }
    category_mapping={
        'Food': 1,
        "Beverages": 1,
        "Fast Food": 1,
        "Budget Meals": 1,
        "Snacks":1,
        'Electronics': 2,
        "Audio Equipment": 2,
        "Mobile Accessories": 2,
        'Clothing': 3,
        "Athletic Footwear": 3,
        "Fashion": 3,
        'Sports': 4,
        "Health": 5,
        "Nutrition": 5,
        "Health Supplements": 5,
        "Fitness Equipment": 5,
        "Beauty": 6,
        "Cosmetics": 6,
        "Skincare": 6,
        "Makeup":6,
        "Vehicle": 7,
        "Home": 8,
        "Home & Garden": 8,
        "Kitchenware": 8,
        "Dorm Essentials": 8,
        "Education": 9,
        "Study Aids": 9,
        "School Supplies": 9,
        "Stationery": 9,
        "Creatives": 9,
        "Kids": 10,
        "Finance": 11,
        "Tech": 12,
        "Technology": 12,
        "HR Software": 12,
        "Project Management Software": 12,
        "Audio Software": 12,
        "Travel": 13,
        "Camping Gear": 13,
        "Travel Gear": 13,
        "Educational":14,
        "Gaming": 15,
        "Gaming Accessories": 15,
        "Entertainment": 16,
        "Investment": 17
    }

    df = convert_age_range_to_avg(df, 'age_range')

     # Apply mapping with dynamic category creation
    df["Platform_mapped"] = df["PlatformUsed"].apply(lambda x: get_or_create_mapping(x, platform_mapping))
    df["ProductCategory_numeric"] = df["ProductCategory"].apply(lambda x: get_or_create_mapping(x, category_mapping))

    df[["Likes", "Shares", "Comments", "Followers"]] = df[["Likes", "Shares", "Comments", "Followers"]].replace({",": ""}, regex=True).astype(float)

    df['SuccessRate'] = ((df['Likes'] + (df['Shares'] * 2) + (df['Comments'] * 1.5)) / df['Followers']) * 100
    df['SuccessRate'] = df['SuccessRate'].clip(0, 100)

    #working on different dataset for trends analysis
    trends_file = os.path.join(config['data']['trends_file'])
    df1 = pd.read_csv(trends_file)
    df1 = convert_age_range_to_avg(df1, 'age_range')

    df = df.sort_values('AgeRange')
    df1 = df1.sort_values('AgeRange')

    print("df columns:", df.columns)
    print("df1 columns:", df1.columns)

    # Merge campaign text with corresponding trends based on age range
    df = pd.merge_asof(df, df1, on='AgeRange', direction='nearest')

    df['trend_score'] = df.apply(lambda row: calculate_similarity(row['CampaignText'], row['Trends'], row['Trends2']), axis=1)

    # Drop trends column as it was only for reference
    df.drop(columns=['Trends', 'Trends2'], inplace=True)

    #emotion probability finder
    classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)
    df_emotions = df['CampaignText'].apply(get_emotion_scores).apply(pd.Series)

    df = pd.concat([df, df_emotions], axis=1)

    # Save processed results to a new CSV file
    output_file = config['data']['raw_data_processed']
    df.to_csv(output_file, index=False)

    print(f"Processing complete! Results saved to {output_file}")

if __name__ == '__main__':
    main()
