# codsoft-internship
🛡️ Credit Card Fraud Detection using Random Forest
This project uses machine learning to detect fraudulent credit card transactions based on real-world transaction data. It applies feature engineering, preprocessing, and a Random Forest classifier to identify fraud with high accuracy.

📁 Dataset
The dataset is split into two files:

fraudTrain.csv — training data

fraudTest.csv — testing data

Each transaction includes details like transaction time, location, merchant, category, and whether it was fraudulent (is_fraud).

⚙️ Features & Workflow
Manual File Selection Uses a file dialog to let users select the CSV files interactively. This avoids path errors and makes the script portable.

Date-Time Feature Engineering Extracts hour and day of week from the transaction timestamp to capture temporal patterns in fraud.

Categorical Encoding Applies LabelEncoder to convert string-based categorical features into numeric format.

Data Scaling Scales numeric features using StandardScaler for better model performance.

Model Training Trains a RandomForestClassifier with class balancing to handle imbalanced fraud data.

Evaluation Prints classification reports for both validation and test sets, including precision, recall, and F1-score.

🧪 How to Run
Make sure you have Python installed and required libraries:

bash
pip install pandas numpy scikit-learn
Then run the script:

bash
python main.py
You’ll be prompted to select the training and testing CSV files manually.

📊 Output
After training, the script prints:

Validation Set Report Performance on a split from the training data.

Test Set Report Performance on unseen test data.

🔍 Future Improvements
Add confusion matrix and ROC curve visualizations

Save trained model for reuse

Try other classifiers like XGBoost or Logistic Regression

Use geolocation features with Haversine distance
