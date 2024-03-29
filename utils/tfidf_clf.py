from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline



def get_best_tfidfclf(X, y):
    # Define a list of classifiers
    classifiers = [
        ('Random Forest', RandomForestClassifier()),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('SVM', SVC()),
        ('Logistic Regression', LogisticRegression()),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Naive Bayes', MultinomialNB())
    ]

    # Use k-fold cross-validation to evaluate each classifier
    best_classifier = None
    best_accuracy = 0

    for name, classifier in classifiers:
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', classifier)
        ])

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

        mean_accuracy = scores.mean()
        print(f'{name} Mean Accuracy: {mean_accuracy:.2f}')

        # Update best classifier if the current one has higher accuracy
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_classifier = (name, classifier)


    best_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', best_classifier[1])
    ])

    return best_model


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

def save_evaluation_results(y_true, y_pred, save_dir, filename):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(y_true, y_pred)

    # Save results to a text file
    save_path = os.path.join(save_dir, f"{filename}.txt")

    with open(save_path, 'w') as file:
        file.write(f'Accuracy: {accuracy:.2f}\n\n')
        file.write('Confusion Matrix:\n')
        file.write(str(conf_matrix) + '\n\n')
        file.write('Classification Report:\n')
        file.write(class_report)

    print(f'Evaluation results saved to: {save_path}')




def run_tfidfclf(df_train, df_test, pp):
    X = df_train['product_text']
    y = df_train['category']
    best_model = get_best_tfidfclf(X, y)
    best_model.fit(X, y)

    # Make predictions on df_test using the best classifier
    y_test = df_test["category"]
    y_test_pred = best_model.predict(df_test['product_text'])
    save_evaluation_results(y_test, y_test_pred, pp.paths["results"], "tfidfclf")

