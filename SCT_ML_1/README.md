# California House Price Prediction (SCT_ML_1)

A Linear Regression model built using the California Housing dataset to predict median house prices based on demographic and geographical features.

---

## 📊 Dataset Description

The California Housing dataset contains data collected from the 1990 California census. The target variable is `House_Price` (representing the median house value in units of $100,000$).

### Features:
1. **MedInc**: Median income in block group (in tens of thousands of dollars).
2. **HouseAge**: Median house age in block group.
3. **AveRooms**: Average number of rooms per household.
4. **AveBedrms**: Average number of bedrooms per household.
5. **Population**: Block group population.
6. **AveOccup**: Average number of household members.
7. **Latitude**: Block group latitude.
8. **Longitude**: Block group longitude.

---

## 📈 Model Performance

After training a Linear Regression model with an **80/20 train-test split** and standardizing the features, the model achieved the following performance metrics on the test set:

- **Mean Squared Error (MSE):** `0.5559`
- **Root Mean Squared Error (RMSE):** `0.7456` ($\approx \$74,560$ deviation on average)
- **R-squared ($R^2$) Score:** `0.5758` (explains $\approx 57.6\%$ of the price variance)

### Feature Coefficients (Importance):
| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
| **MedInc** | `0.8544` | High positive correlation; higher median income significantly increases house value. |
| **AveBedrms** | `0.3393` | Positive correlation with the average number of bedrooms. |
| **HouseAge** | `0.1225` | Minor positive correlation; older houses tend to have higher value in this dataset. |
| **Population** | `-0.0023` | Negligible impact. |
| **AveOccup** | `-0.0408` | Minor negative correlation. |
| **AveRooms** | `-0.2944` | Negative correlation (controlling for bedrooms). |
| **Longitude** | `-0.8698` | Negative geographical correlation (westward location). |
| **Latitude** | `-0.8969` | Negative geographical correlation (northward location). |

---

## 🛠️ How to Run

1. Navigate to this directory:
   ```bash
   cd SCT_ML_1
   ```
2. Run the training script:
   ```bash
   python housing_regression.py
   ```
