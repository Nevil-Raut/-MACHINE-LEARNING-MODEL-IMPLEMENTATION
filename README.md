# -MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: RAUT NEVIL VIJAY

*INTERN ID*: CT04DY1650

*DOMAIN*: Python Programming

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

The provided Python script is a comprehensive workflow for building a machine learning-based spam detection system using SMS messages. It begins by importing essential libraries for data handling, machine learning, and visualization, including `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`. The script then sets up a plotting style and ensures the visualization environment is ready. The dataset, sourced from the UCI Machine Learning Repository, is automatically downloaded and extracted if not already present. It consists of SMS messages labeled as "ham" (non-spam) or "spam," which are loaded into a pandas DataFrame. The script provides an overview of the dataset, including the total number of messages, the distribution of spam versus ham messages, and basic descriptive statistics of message lengths. Class distributions are visualized using bar and pie charts, highlighting the imbalance between spam and ham messages, and histograms and boxplots reveal differences in message lengths, showing that spam messages tend to be longer on average. Missing values are checked, and the labels are converted into binary numerical form, where ham is represented by 0 and spam by 1, preparing the data for model training.

The dataset is split into training and testing subsets with stratification to maintain class proportions, and sample messages from both sets are displayed for reference. A machine learning pipeline is created, combining `TfidfVectorizer` for converting textual messages into numerical features and `LogisticRegression` for classification. The TF-IDF vectorizer is configured with options like lowercase conversion, English stop words removal, n-grams up to two words, and feature limits. Before hyperparameter tuning, the vectorizer is fitted to the training data to examine the feature matrix and extract sample features. A grid search over parameters such as TF-IDF maximum document frequency, n-gram ranges, regularization strength, and penalty type is performed using five-fold cross-validation to optimize the F1 score, resulting in the selection of the best-performing model.

Once the model is trained, predictions are made on the test set, and the performance is evaluated using a classification report, confusion matrix, ROC curve, and AUC score. The confusion matrix is visualized, and the ROC curve is plotted to show the trade-off between true positive and false positive rates. The model’s coefficients are analyzed to identify the most influential features in spam detection, and a bar chart highlights the top 15 indicators, where positive coefficients correspond to spam signals and negative coefficients indicate ham signals. The trained model is then tested on a set of example messages, demonstrating its ability to classify new messages correctly, and prediction probabilities provide insights into confidence levels. Finally, the model is saved using `joblib` for future use and verified by loading it back and making a sample prediction. This pipeline provides a full end-to-end solution for spam detection, encompassing data preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, feature importance analysis, and deployment readiness, making it a robust example of text classification using Python and machine learning.

This description is approximately 500 words.

If you want, I can also rewrite it in a **more concise, reader-friendly storytelling style** for reports or presentations. Do you want me to do that?

<img width="869" height="438" alt="Image" src="https://github.com/user-attachments/assets/a4cf5a1d-e636-4557-8a95-631ad30c3bf2" />

<img width="542" height="516" alt="Image" src="https://github.com/user-attachments/assets/49d19952-27f5-495b-bb10-f2dda3fdeace" />

<img width="621" height="407" alt="Image" src="https://github.com/user-attachments/assets/30911730-ba21-429c-9f82-0d68d928a360" />

<img width="558" height="546" alt="Image" src="https://github.com/user-attachments/assets/809ce72a-b982-4598-908d-e7506b43a311" />

<img width="573" height="425" alt="Image" src="https://github.com/user-attachments/assets/a454e5c0-11d4-4f19-bc8e-cd63bead25c5" />

<img width="253" height="514" alt="Image" src="https://github.com/user-attachments/assets/1ad93897-2489-4fe9-8fa6-b04e44153aa2" />

<img width="856" height="548" alt="Image" src="https://github.com/user-attachments/assets/3c7229f2-13c2-43c8-a988-b615710f3b02" />

<img width="692" height="151" alt="Image" src="https://github.com/user-attachments/assets/136b81c7-f4c3-4f23-8f57-fd5661ee282e" />
