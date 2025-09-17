# Two-Tower Recommender System

A **hybrid two-tower recommender system** for personalized product recommendations. This project implements a deep learning-based recommendation model that leverages both **user features** and **item features** using a two-tower architecture, enabling recommendations for both existing and new users.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Getting Started](#getting-started)

   * [Installation](#installation)
   * [Dataset & Artifacts](#dataset--artifacts)
5. [Training](#training)
6. [Inference & Recommendations](#inference--recommendations)

   * [Existing Users](#existing-users)
   * [New Users](#new-users)
   * [Predict Rating](#predict-rating)
7. [Project Structure](#project-structure)
8. [Future Improvements](#future-improvements)
9. [License](#license)

---

## Project Overview

This project implements a **Two-Tower Hybrid Recommender System**:

* **User Tower:** Encodes user features (categorical and numerical).
* **Item Tower:** Encodes item features (categorical).
* **Interaction:** Computes similarity between user and item embeddings (dot product or cosine similarity) to predict ratings.

The system supports:

* Personalized recommendations for **existing users**
* Recommendations for **new users** based on profile features
* **Rating prediction** for any user-item pair

---

## Key Features

* Two-tower deep learning architecture implemented in **PyTorch**
* Handles **categorical and numerical features**
* Negative sampling for robust training
* Cosine similarity or dot-product for interaction scoring
* Efficient **batch inference** for recommendations
* Works for both **existing and new users**
* Saves preprocessing artifacts and model for standalone deployment

---

## Architecture

```
User Features          Item Features
  (categorical + num)       (categorical)
        |                       |
        |--- Embeddings --------|
        |                       |
     User Tower               Item Tower
        |                       |
       Dense                    Dense
        |                       |
        -------- Interaction ----
                 |
            Predicted Rating
```

* User tower combines embeddings for categorical features + normalized numerical features.
* Item tower combines embeddings for categorical features.
* Interaction layer outputs predicted rating via **cosine similarity + sigmoid**.

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/two-tower-recommender.git
cd two-tower-recommender

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset & Artifacts

* **Data:** Preprocessed transactional dataset (`df_clean`)
* **Preprocessing artifacts:** Stored in `preproc_artifacts.pkl`
* **Recommendation assets:** Stored in `recommendation_assets/recommendation_assets.pkl`

Artifacts include:

* Feature mappings for categorical columns
* Scalers for numerical columns
* User profiles and purchase history
* Item catalog

---

## Training

1. Split dataset into **train / validation / test** by date.
2. Encode categorical columns and scale numerical columns.
3. Use `TwoTowerDataset` for **training and negative sampling**.
4. Train `TwoTowerHybrid` model using **binary cross-entropy loss**, early stopping, and learning rate scheduler.
5. Save best model and preprocessing artifacts:

```python
model = train_two_tower_model(train_dataset, val_dataset, user_cat_sizes, item_cat_sizes, user_num_cols)
torch.save(model.state_dict(), "best_two_tower_model.pth")
```

---

## Inference & Recommendations

### Existing Users

```python
recommender = StandaloneRecommender(
    model_path="best_two_tower_model.pth",
    artifacts_path="preproc_artifacts.pkl",
    assets_path="recommendation_assets",
    user_cat_cols=user_cat_cols,
    user_num_cols=user_num_cols,
    item_cat_cols=item_cat_cols,
    device='cpu'
)

recommendations = recommender.recommend_for_user(customer_id=12345, top_k=5)
```

* Filters out previously purchased items (optional).
* Returns top-K items with predicted rating.

### New Users

```python
new_user = {
    'Age': 28, 'Gender': 'Female', 'Country': 'UK',
    'City': 'London', 'State': 'England', 'Income': 'High',
    'Customer_Segment': 'Premium', 'Feedback': 'Excellent',
    'Shipping_Method': 'Express', 'Payment_Method': 'Credit Card',
    'Order_Status': 'Delivered', 'Total_Purchases': 3, 'Amount': 300.0
}

recommendations = recommender.recommend_for_new_user(new_user, top_k=5)
```

* Generates recommendations purely based on user profile features.

### Predict Rating

```python
item_features = {
    'products': 'iPhone 14',
    'Product_Category': 'Electronics',
    'Product_Brand': 'Apple',
    'Product_Type': 'Smartphone'
}

rating = recommender.predict_rating(new_user, item_features)
print(f"Predicted rating: {rating:.2f}/5.0")
```

---

## Project Structure

```
├── training.py                  # Training scripts for TwoTowerHybrid model
├── standalone_recommender.py    # Pretrained recommender for inference
├── preproc_artifacts.pkl        # Preprocessing mappings & scalers
├── recommendation_assets/       # User profiles, item catalog, histories
├── best_two_tower_model.pth     # Trained PyTorch model
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Future Improvements

* Implement **personalized ranking loss** (BPR or pairwise loss)
* Support **real-time online updates** for user/item embeddings
* Improve hyperparameter tuning for **better generalization**

---
