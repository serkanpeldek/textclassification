# Text Classification with TF-IDF + CLF and Hugging Face DistileBert

## Preprocessing Overview:

In the process of preparing the text data for category prediction, a series of preprocessing steps were implemented to ensure the quality and consistency of the input. Each step in the pipeline serves a specific purpose in refining the textual information provided by sellers. Here is a brief overview of the preprocessing steps:

1. **MakeString:**
   - Conversion to a string format ensures uniformity in handling different types of input, promoting a consistent data format.

2. **ReplaceBy:**
   - The replacement of specified characters or patterns with others helps to eliminate potential noise or irrelevant symbols that might not contribute to meaningful information.

3. **LowerText:**
   - Standardizing all text to lowercase aids in achieving uniformity and removes potential discrepancies due to letter case.

4. **ReduceTextLength:**
   - Truncating text to a specified length is particularly useful for addressing computational constraints and promoting efficiency during processing.

5. **VectorizeText:**
   - Breaking down the text into individual words facilitates subsequent analysis and enables the extraction of meaningful patterns.

6. **FilterPunctuation:**
   - Eliminating punctuation is crucial for focusing on the core content of the text and avoiding unnecessary variations introduced by punctuation marks.

7. **FilterNonalpha:**
   - Removing non-alphabetic words ensures that only words composed of letters are considered, contributing to the extraction of meaningful linguistic patterns.

8. **FilterStopWord:**
   - The exclusion of common stopwords in the Turkish language helps in reducing the dimensionality of the data and focusing on words with more semantic significance.

9. **FilterShortWord:**
   - Eliminating short words contributes to a more refined representation of the language, excluding potentially irrelevant or ambiguous terms.

10. **TextProcessor:**
    - The combination of these preprocessing steps into a unified pipeline ensures a systematic and standardized approach to text transformation.

11. **JoinWithSpace:**
    - Reassembling the processed words into a coherent string with spaces maintains the structure necessary for subsequent analysis.

These preprocessing steps collectively contribute to creating a cleaner, more focused representation of the text data. The rationale behind each step is to eliminate noise, enhance interpretability, and improve the overall performance of the models developed for category prediction. The resulting processed text serves as the foundation for building robust and accurate models in the subsequent stages of the analysis.

## Part A: Model Development and Comparison

### Method 1: TF-IDF + Classifier

**Methodology:**
- Utilized TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical vectors.
- Employed a classifier (algorithm not specified) for category prediction.

**Results:**
- Training Accuracy: 75%
- Testing Accuracy: 74%

**Pros and Cons:**
- *Pros:* Classic and computationally efficient, interpretability with TF-IDF scores.
- *Cons:* Limited in capturing complex relationships, may struggle with semantic meaning.

### Method 2: DistilBERT

**Methodology:**
- Leveraged DistilBERT, a pre-trained transformer-based model, for contextualized embeddings.

**Results:**
- Training Accuracy: 85%
- Testing Accuracy: 77%

**Pros and Cons:**
- *Pros:* Captures complex relationships, pre-trained on a large corpus.
- *Cons:* Requires more computational resources, less interpretable.

### Comments

- Due to constraints in computational resources and time, a comprehensive hyperparameter search wasn't possible, potentially limiting the models' performance.
- Model deployment considerations and integration into a production environment were not discussed.

## Part B: Unsupervised Category Creation

**Methodology:**
- Applied TF-IDF for text representation and used KMeans clustering for unsupervised category creation.

**Results:**
- 5 cluster used to assign prdoct text



## General Comments:

- The dataset was subsampled to address class imbalance issues, ensuring a balanced representation of categories in the training set.
- Additional preprocessing steps such as data cleaning, handling missing values, or addressing imbalanced classes weren't explicitly mentioned.
- Code snippets or key portions of the code were not provided for a detailed understanding of the implementation.
- Consideration of model interpretability, potential challenges, and deployment aspects would enhance the comprehensiveness of the report.
- The report lacks a clear presentation structure. Consider organizing sections more explicitly for better readability.

**Conclusion:**
Despite limitations, both TF-IDF + Classifier and DistilBERT show promise for category prediction. The choice between them depends on the trade-off between interpretability and performance. Further exploration with increased computational resources could yield more refined models.


