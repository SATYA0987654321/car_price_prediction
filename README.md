# 🚗 Used Car Price Prediction System (Indian Automotive Market)

A simple, functional, and practical machine learning application that predicts the price of used cars in India based on vehicle specifications.

Built with **Python, Pandas, Scikit-learn, XGBoost, and Streamlit**.

---

## 📁 Project Structure

```
Car_Price_Prediction/
│
├── car_data.csv            # Indian used cars dataset (Maruti, Tata, Mahindra, etc.)
├── train_model.py          # Script to train and evaluate ML models
├── car_price_model.pkl     # Saved best trained model (XGBoost) & metrics
├── app.py                  # Streamlit web application
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
```

---

## 🇮🇳 Indian Cars Supported

- **Maruti Suzuki**: Swift, Baleno, Dzire, Brezza, Ertiga, Alto, Wagon R
- **Tata**: Nexon, Punch, Harrier, Safari, Tiago, Altroz
- **Mahindra**: Thar, Scorpio-N, XUV700, Bolero, XUV300
- **Hyundai**: Creta, Venue, i20, Verna, Grand i10
- **Kia**: Seltos, Sonet, Carens
- **Toyota**: Innova Crysta, Fortuner, Glanza, Urban Cruiser
- **Honda**: City, Amaze
- **Volkswagen**: Taigun, Virtus, Polo
- **Luxury Segments**: BMW (3 Series, X1), Mercedes-Benz (A-Class, C-Class)

**Fuel Types Supported:** Petrol, Diesel, CNG, Electric, Hybrid.  
**Valuation Format:** Indian Rupees (₹ in Lakhs & Crores).

---

## 📊 Model Evaluation Results

Models trained on an 80/20 train/test split:

| Model | MAE (₹) | RMSE (₹) | R² Score |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor** | **₹38,172** | **₹58,347** | **0.9912 (99.1%)** 🏆 |
| **Random Forest Regressor** | ₹42,373 | ₹69,281 | 0.9876 (98.8%) |
| **Linear Regression** | ₹1,04,613 | ₹1,85,017 | 0.9113 (91.1%) |

*XGBoost achieved the highest R² score and lowest error, so it is used as the default engine for price predictions.*

---

## 🚀 How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train models (optional, already trained):
```bash
python train_model.py
```

### 3. Launch Streamlit App:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
