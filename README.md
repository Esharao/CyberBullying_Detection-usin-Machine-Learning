# CyberBullying Detection using Machine Learning

This project is a web-based application built with Flask that detects cyberbullying in user-submitted text using a machine learning model.

1. Application Structure
   Flask Application: The application is set up with Flask, a Python micro-framework, to create a web server and handle HTTP requests.
   Model and Vectorizer: A machine learning model (stored in LinearSVCTuned.pkl) and a TF-IDF vectorizer (tfidfvectoizer.pkl) are pre-trained to classify text. The vectorizer processes input text, converting it 
   into numerical form for the model.
2. Files Used
3. stopwords.txt: Contains stop words (common words that add little value in NLP) that the vectorizer filters out.
   LinearSVCTuned.pkl: A pre-trained SVM model for classification.
   tfidfvectoizer.pkl: A pre-trained TF-IDF vectorizer to convert text into numerical data suitable for the model.
   AdaBoostClassifier.pkl: An ensemble model that combines multiple weak learners (often decision trees) to boost overall classification performance by adjusting weights based on previous errors.
   DecisionTreeClassifier.pkl: A tree-based model that splits data into branches to make decisions based on feature values, creating interpretable rules for classification.
   LinearSVC.pkl: A linear Support Vector Classifier (SVC) that finds a hyperplane to separate classes in high-dimensional space, particularly effective for text classification.
   LogisticRegression.pkl: A regression-based classifier that models the probability of class membership using a logistic function, useful for binary classification tasks.
   SGDClassifier.pkl: A linear model that uses Stochastic Gradient Descent, well-suited for large datasets and real-time applications, especially for text data.
   BaggingClassifier.pkl: An ensemble model that uses bootstrap aggregating (bagging) to train multiple base models (e.g., decision trees) independently, enhancing model stability and accuracy.

5. HTML (Front-End)
   Form and Result Display:
A form allows users to input text.
   Depending on the model’s prediction, the result is shown as "Bullying" or "Non-Bullying," with different background colors.
6. Bootstrap and CSS
   Bootstrap Integration: The application includes Bootstrap for responsive design.
   CSS Styling: Custom styles add a professional look to the page. The color coding in .bullying (red) and .non-bullying (blue) enhances the clarity of the result.
7. How to Run Locally
   Install dependencies: Flask, scikit-learn, and pickle.
   Place stopwords.txt, tfidfvectoizer.pkl, and LinearSVCTuned.pkl in the project directory.
   Run the app: python app.py and open localhost:5000 in your browser.
8. Potential Enhancements
   Error Handling: Add checks to handle empty input or server errors.
   Model Improvement: Experiment with other NLP techniques or models for improved accuracy.
