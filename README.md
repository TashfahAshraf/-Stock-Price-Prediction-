# 📈 Stock Price Prediction

## 📌 Author  
👩‍💻 **Tashfah Ashraf**  
*Intern – ARCH TECHNOLOGIES*  

---

## 📌 Task Overview  

This project was developed as part of my internship.  
The task was to **predict future stock prices** based on historical stock data using **Machine Learning (ML)** and **Deep Learning (DL)** techniques.  

We were provided with a dataset containing features such as:  
- **Open Price**  
- **High Price**  
- **Low Price**  
- **Closing Price**  
- **Trading Volume**  

The objective was to build models that can learn from this historical data and **predict the next-day closing price**.  

---

## 🔧 Steps Performed  

### 1️⃣ Data Preprocessing  
- Loaded stock price dataset (from Kaggle).  
- Converted `Date` to datetime format and sorted chronologically.  
- Cleaned `Volume` (converted to numeric, removed missing values).  
- Filtered data for a single stock (**AAPL**).  
- Created Target column: **Close_t+1 = next day’s closing price**.  
- Performed train-test split (80% train, 20% test) using time-based split.  

### 2️⃣ Baseline Model: Linear Regression  
- Used features: **Open, High, Low, Close, Volume**.  
- Predicted `Close_t+1`.  
- Evaluated using:  
  - 📉 **Mean Absolute Error (MAE)**  
  - 📉 **Root Mean Squared Error (RMSE)**  
  - 📊 **R² Score**  
- Visualized **Actual vs Predicted Prices**.  

### 3️⃣ Advanced Model: LSTM (Long Short-Term Memory)  
- Scaled features with **MinMaxScaler**.  
- Created time sequences (lookback = **60 days**).  
- Built LSTM model with:  
  - 2 LSTM layers  
  - Dropout for regularization  
  - Dense output layer  
- Trained with **EarlyStopping** to avoid overfitting.  
- Predicted next-day closing prices.  
- Evaluated with MAE, RMSE, R².  
- Visualized **Actual vs Predicted Prices (LSTM)**.  

### 4️⃣ Naïve Baseline  
- Predicted **tomorrow’s Close = today’s Close**.  
- Served as a **benchmark** to compare ML/DL models.  

### 5️⃣ Model Comparison  
<img width="547" height="152" alt="image" src="https://github.com/user-attachments/assets/3a282f52-2941-4a39-a7b6-14a37a5ebce0" />  

---

## 📊 Results & Conclusion  

- ✅ **Linear Regression** performed better than the naïve baseline, showing ML can capture basic trends.  
- ✅ **LSTM outperformed Linear Regression**, achieving the lowest errors, because it captures temporal dependencies in stock data.  
- 🎯 This demonstrates that **Deep Learning (LSTM)** is more suitable for stock price forecasting compared to simple regression.  

---

## 🛠 Tech Stack  

- **Python 3**  
- **Pandas, NumPy** – data preprocessing  
- **Matplotlib** – visualization  
- **Scikit-learn** – Linear Regression, metrics, scaling  
- **TensorFlow / Keras** – LSTM model  

---

## 🚀 How to Run  

### 🔹 Clone this repo:
git clone <your-repo-link>
cd stock-price-prediction

###  🔹Install required libraries:
pip install -r requirements.txt

### 🔹 Run the Jupyter Notebook:
jupyter notebook StockPrice.ipynb
