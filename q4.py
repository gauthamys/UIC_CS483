import numpy as np
import pandas as pd

# Step 1: Load the data
R = np.loadtxt('q4/data/user-shows.txt')
with open('q4/data/shows.txt', 'r') as f:
    shows = [line.strip() for line in f]

# Step 2: Compute matrices P and Q
P = np.diag(np.sum(R, axis=1))  # P is a diagonal matrix with user degrees
Q = np.diag(np.sum(R, axis=0))  # Q is a diagonal matrix with item degrees

# Step 3: Compute recommendation matrix for user-user collaborative filtering
P_inv_sqrt = np.diag(1 / np.sqrt(np.diag(P)))  # P^(-1/2)
P_inv_sqrt[np.isinf(P_inv_sqrt)] = 0  # Handle division by zero

# User-user similarity matrix
S_U = P_inv_sqrt @ R @ R.T @ P_inv_sqrt

# Recommendation matrix for user-user collaborative filtering
Gamma_user_user = S_U @ R

# Step 4: Find top 5 recommendations for Alex (user 499) for the first 100 shows
alex_scores_user_user = Gamma_user_user[499, :100]
top_5_indices_user_user = np.argsort(-alex_scores_user_user)[:5]
top_5_shows_user_user = [shows[i] for i in top_5_indices_user_user]

print("Top 5 recommendations for Alex using user-user collaborative filtering:")
print(top_5_shows_user_user)

# Step 5: Compute recommendation matrix for item-item collaborative filtering
Q_inv_sqrt = np.diag(1 / np.sqrt(np.diag(Q)))  # Q^(-1/2)
Q_inv_sqrt[np.isinf(Q_inv_sqrt)] = 0  # Handle division by zero

# Item-item similarity matrix
S_I = Q_inv_sqrt @ R.T @ R @ Q_inv_sqrt

# Recommendation matrix for item-item collaborative filtering
Gamma_item_item = R @ S_I

# Step 6: Find top 5 recommendations for Alex (user 499) for the first 100 shows
alex_scores_item_item = Gamma_item_item[499, :100]
top_5_indices_item_item = np.argsort(-alex_scores_item_item)[:5]
top_5_shows_item_item = [shows[i] for i in top_5_indices_item_item]

print("\nTop 5 recommendations for Alex using item-item collaborative filtering:")
print(top_5_shows_item_item)
