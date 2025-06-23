import pandas as pd
import json # To output results in JSON format for easier integration
from collections import Counter
import warnings

# --- Configuration ---
LOG_FILE = 'emotion_log.csv'
CONFIDENCE_THRESHOLD = 0.6 # Ignore predictions below this confidence level (e.g., 60%)

# Define emotion categories (adjust based on your interpretation)
POSITIVE_EMOTIONS = ['Happy']
NEGATIVE_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Sad']
NEUTRAL_EMOTIONS = ['Neutral', 'Surprise'] # Grouping Surprise as Neutral/Ambiguous here

# --- Analysis Functions ---

def analyze_overall_sentiment(df):
    """Calculates overall sentiment distribution."""
    total_detections = len(df)
    if total_detections == 0:
        return {"total_detections": 0, "sentiment_counts": {}, "sentiment_percentages": {}}

    def categorize_emotion(emotion):
        if emotion in POSITIVE_EMOTIONS:
            return 'Positive'
        elif emotion in NEGATIVE_EMOTIONS:
            return 'Negative'
        elif emotion in NEUTRAL_EMOTIONS:
            return 'Neutral'
        else:
            return 'Other' # Should not happen with default labels

    df['SentimentCategory'] = df['Emotion'].apply(categorize_emotion)
    sentiment_counts = df['SentimentCategory'].value_counts().to_dict()
    sentiment_percentages = (df['SentimentCategory'].value_counts(normalize=True) * 100).round(2).to_dict()

    return {
        "total_valid_detections": total_detections,
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": sentiment_percentages
    }

def analyze_emotion_counts(df):
    """Counts occurrences of each specific emotion."""
    if len(df) == 0:
        return {}
    return df['Emotion'].value_counts().to_dict()

def analyze_by_location(df):
    """Performs analysis grouped by camera location."""
    results = {}
    if 'CameraLocation' not in df.columns:
        print("Warning: 'CameraLocation' column not found in CSV. Skipping location analysis.")
        return results
    if len(df) == 0:
        return results

    grouped = df.groupby('CameraLocation')
    for location, group_df in grouped:
        loc_overall = analyze_overall_sentiment(group_df.copy()) # Use copy to avoid SettingWithCopyWarning
        loc_emotions = analyze_emotion_counts(group_df)
        results[location] = {
            "overall_sentiment": loc_overall,
            "emotion_counts": loc_emotions
        }
    return results

def analyze_by_hour(df):
    """Performs analysis grouped by hour of the day."""
    results = {}
    if 'Timestamp' not in df.columns or len(df) == 0:
        print("Warning: 'Timestamp' column not found or empty DataFrame. Skipping hourly analysis.")
        return {}

    # Ensure Timestamp is datetime
    try:
        # Infer datetime format, handle potential errors
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True) # Drop rows where conversion failed
        if len(df) == 0: return results # Check again after dropping NaT
    except Exception as e:
        print(f"Warning: Could not parse 'Timestamp' column effectively: {e}. Skipping hourly analysis.")
        return {}


    df['Hour'] = df['Timestamp'].dt.hour
    grouped = df.groupby('Hour')
    for hour, group_df in grouped:
        hour_overall = analyze_overall_sentiment(group_df.copy()) # Use copy
        hour_emotions = analyze_emotion_counts(group_df)
        results[f"{hour:02d}:00 - {hour:02d}:59"] = { # Format hour nicely
            "overall_sentiment": hour_overall,
            "emotion_counts": hour_emotions
        }
    return results


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Analyzing emotion log: {LOG_FILE}")
    print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")

    # Suppress SettingWithCopyWarning for cleaner output in this script
    warnings.simplefilter(action='ignore', category=FutureWarning) # For potential future pandas changes
    #pd.options.mode.chained_assignment = None  # Be cautious using this globally

    try:
        # Read CSV, explicitly parse timestamps
        df = pd.read_csv(LOG_FILE)

        # Basic data cleaning and preparation
        if 'Confidence' not in df.columns:
             raise ValueError("CSV file must contain a 'Confidence' column.")
        df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
        df.dropna(subset=['Confidence', 'Emotion'], inplace=True) # Drop rows with invalid confidence or emotion

        print(f"Read {len(df)} rows initially.")

        # Apply confidence filter
        df_filtered = df[df['Confidence'] >= CONFIDENCE_THRESHOLD].copy() # Use copy after filtering
        print(f"Using {len(df_filtered)} rows after applying confidence threshold >= {CONFIDENCE_THRESHOLD}")

        if len(df_filtered) == 0:
            print("No data meets the confidence threshold. Exiting analysis.")
            analysis_results = {"error": "No data meets threshold"}
        else:
             # Perform analyses
            overall_sentiment = analyze_overall_sentiment(df_filtered)
            emotion_counts = analyze_emotion_counts(df_filtered)
            location_analysis = analyze_by_location(df_filtered)
            hourly_analysis = analyze_by_hour(df_filtered) # Pass the filtered df

            # Combine results into a single dictionary
            analysis_results = {
                "metadata": {
                    "log_file": LOG_FILE,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "total_rows_read": len(df),
                    "rows_analyzed": len(df_filtered)
                },
                "overall_sentiment_summary": overall_sentiment,
                "overall_emotion_counts": emotion_counts,
                "analysis_by_location": location_analysis,
                "analysis_by_hour": hourly_analysis
            }

        # --- Output results as JSON ---
        # This makes it easy for Node-RED (or other programs) to consume
        json_output = json.dumps(analysis_results, indent=4)
        print("\n--- Analysis Results (JSON) ---")
        print(json_output)
        print("--- End of Analysis ---")


    except FileNotFoundError:
        print(f"Error: Log file '{LOG_FILE}' not found.")
        # Output minimal JSON error
        print(json.dumps({"error": f"Log file '{LOG_FILE}' not found."}, indent=4))
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        # Output minimal JSON error
        print(json.dumps({"error": f"Analysis failed: {e}"}, indent=4))