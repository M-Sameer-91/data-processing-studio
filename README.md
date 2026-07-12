# Data Processing Studio
A desktop GUI application built with Python + Tkinter that walks a dataset through the full preprocessing pipeline used in classical Machine Learning — from raw CSV to a clean, encoded, scaled dataset ready for modeling — all through a point-and-click interface (no code required to operate it).


📊 Data Processing & Visualization Studio

A desktop GUI application built with Python + Tkinter that walks a dataset through the full preprocessing pipeline used in classical Machine Learning — from raw CSV to a clean, encoded, scaled dataset ready for modeling — all through a point-and-click interface (no code required to operate it).

Built as a hands-on project while learning core ML data preprocessing concepts: missing values, categorical encoding, feature scaling, and exploratory statistics/visualization.


✨ Features

📂 Upload & Preview


Load .csv, .data, .json, or .xlsx files
Optional custom column headers
Instant data preview (head) and full dataset info (shape, dtypes, non-null counts)


🧹 Process Data (Missing Values)


Replace placeholder characters (e.g. ?) with NaN
Missing value summary table (column, count, dtype)
Impute missing values using mean, median, most frequent, or drop affected rows
Convert column data types (float, int, str)
Export the cleaned dataset as a new CSV


🏷️ Encoding


Label Encoding (sklearn.LabelEncoder) for binary/ordinal columns — shows the class → number mapping
One-Hot Encoding via either:

pandas.get_dummies
sklearn.OneHotEncoder



Original column is automatically dropped and replaced with the new encoded columns


⚖️ Feature Scaling


Train/test split (sklearn.train_test_split) with configurable test size, random state, and shuffle
Apply StandardScaler or MinMaxScaler to selected numeric columns
Scaler is fit only on X_train and applied to both X_train/X_test to avoid data leakage
Preview and describe() on X_train, X_test, y_train, y_test at any point


📈 Statistics


describe() summary — numeric-only or all columns


📊 Visualization


Histogram with mean/median reference lines for any numeric column
IQR-based outlier detection with a detailed report


🕘 Activity Log


Every action (load, impute, encode, scale, plot) is logged live at the bottom of the app



🛠️ Tech Stack


Python 3
Tkinter / ttk — GUI
pandas, NumPy — data handling
scikit-learn — SimpleImputer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, train_test_split
Matplotlib — histograms embedded directly in the Tkinter window



🚀 How to Run


Clone the repo


bash   git clone https://github.com/<your-username>/data-processing-studio.git
   cd data-processing-studio


Install dependencies


bash   pip install pandas numpy scikit-learn matplotlib

(Tkinter ships with most standard Python installations; on Linux you may need sudo apt install python3-tk)


Run the app


bash   python data_processing_studio.py


Use it

Go to Upload & Preview → Browse → select a CSV/Excel/JSON file → Load Dataset
Move through the tabs in order: Process Data → Encoding → Feature Scaling → Statistics → Visualization






📸 Screenshots

(Add your screenshots here — e.g.)

Upload & PreviewEncodingFeature ScalingShow ImageShow ImageShow Image


📌 Project Motivation

This project was built to reinforce core data preprocessing concepts for Machine Learning by turning notebook-based exercises (missing value handling, categorical encoding, feature scaling) into a single interactive, reusable tool — rather than re-running the same boilerplate code in a new notebook for every dataset.


🔮 Possible Future Improvements


Support for multi-column scaling presets / saved pipelines
Export the fitted encoders/scalers (via joblib) for reuse in a downstream model
Correlation heatmap and pairplot in the Visualization tab
Undo/redo for preprocessing steps
Dark mode
