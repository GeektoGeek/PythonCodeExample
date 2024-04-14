"""import pymongo
import json

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27023/")
db = client["ListingandReviews"]
collection = db["listingandreview"]

# Retrieve records from MongoDB
cursor = collection.find({})

# Process records as JSON objects
for document in cursor:
    # Convert MongoDB document to JSON
    json_document = json.dumps(document)
    
    # Process JSON document (for example, print it)
    print(json_document)
    
    # Perform further operations with the JSON document
    # For example, you can parse the JSON document and extract specific fields, manipulate the data, etc.

"""
import pymongo
import json
import datetime
import decimal
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import word_tokenize  # Import for tokenization

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def convert_to_string(value):
    if not is_json_serializable(value):
        if isinstance(value, datetime.datetime):
            value = value.isoformat()  # Convert to ISO 8601 format
        else:
            value = str(value)  # Convert non-serializable values to string
    return value


def process_document(document):
    previous_datetime = None  # Initialize variable for previous datetime
    for key, value in document.items():
        if key == "datetime":  # Store previous datetime for later use
            previous_datetime = convert_to_string(value)
        document[key] = convert_to_string(value)  # Convert all non-serializable values
    return document, previous_datetime


# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27023/")
db = client["ListingandReviews"]
collection = db["listingandreview"]

# Retrieve records from MongoDB
cursor = collection.find({})

# Process records as JSON objects and store in a list
processed_documents = []
for result in cursor:
    try:
        document, prev_datetime = process_document(result.copy())
        json_document = json.dumps(document)
        processed_documents.append(json_document)
    except Exception as e:
        print(f"Error processing document: {e}")

# Close the connection
client.close()

# Convert the list of JSON strings to a DataFrame
df = pd.DataFrame(json.loads(doc) for doc in processed_documents)

# **Define functions for smaller analysis steps**

def tokenize_summary(summary):
    """Tokenizes the summary text, converting it to lowercase."""
    try:
        tokens = word_tokenize(summary.lower())
        return tokens
    except Exception as e:
        print(f"Error tokenizing summary: {e}")
        return []  # Return an empty list on error


def remove_stop_words(tokens):
    """Removes common stop words from the tokenized text (optional)."""
    from nltk.corpus import stopwords  # Import inside the function
    stop_words = stopwords.words('english')
    return [word for word in tokens if word not in stop_words]


def analyze_summary(summary):
    """Performs NLP analysis on a summary.

    This function calls individual analysis functions and combines the results.
    """
    tokens = tokenize_summary(summary)

    # ... (Add other analysis steps with separate functions and error handling)

    # Combine and return analysis results
    analysis_results = {
        "tokens": tokens,
        # ... (add results from other analysis steps)
    }
    return analysis_results


# **Apply analysis to the 'summary' column with error handling**

try:
    df["analysis"] = df["summary"].apply(analyze_summary)
except Exception as e:
    print(f"Error applying analysis to summaries: {e}")

# Print the DataFrame with the analysis results
#print(df.to_string())