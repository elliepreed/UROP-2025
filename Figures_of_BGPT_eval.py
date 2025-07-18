import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Step 1: Create the data ---

data = [
    # Revision 0
    {"revision": 0, "model": "Simultaneous", "language": "English", "accuracy": 50.00},
    {"revision": 0, "model": "Simultaneous", "language": "Dutch", "accuracy": 53.37},
    {"revision": 0, "model": "Sequential", "language": "English", "accuracy": 58.70},
    {"revision": 0, "model": "Sequential", "language": "Dutch", "accuracy": 58.34},

    # Revision 64200
    {"revision": 64200, "model": "Simultaneous", "language": "English", "accuracy": 94.81},
    {"revision": 64200, "model": "Simultaneous", "language": "Dutch", "accuracy": 73.06},
    {"revision": 64200, "model": "Sequential", "language": "English", "accuracy": 94.03},
    {"revision": 64200, "model": "Sequential", "language": "Dutch", "accuracy": 74.56},

    # Revision 64300
    {"revision": 64300, "model": "Simultaneous", "language": "English", "accuracy": 94.42},
    {"revision": 64300, "model": "Simultaneous", "language": "Dutch", "accuracy": 75.33},
    {"revision": 64300, "model": "Sequential", "language": "English", "accuracy": 94.42},
    {"revision": 64300, "model": "Sequential", "language": "Dutch", "accuracy": 78.16},

    # Revision 67000
    {"revision": 67000, "model": "Simultaneous", "language": "English", "accuracy": 94.03},
    {"revision": 67000, "model": "Simultaneous", "language": "Dutch", "accuracy": 88.03},
    {"revision": 67000, "model": "Sequential", "language": "English", "accuracy": 86.75},
    {"revision": 67000, "model": "Sequential", "language": "Dutch", "accuracy": 90.73},

    # Revision 128000
    {"revision": 128000, "model": "Simultaneous", "language": "English", "accuracy": 94.80},
    {"revision": 128000, "model": "Simultaneous", "language": "Dutch", "accuracy": 94.34},
    {"revision": 128000, "model": "Sequential", "language": "English", "accuracy": 84.55},
    {"revision": 128000, "model": "Sequential", "language": "Dutch", "accuracy": 95.41},
]

df = pd.DataFrame(data)

# Save to CSV (optional)
df.to_csv("model_scores.csv", index=False)
print("✅ Saved data to model_scores.csv")

# --- Step 2: Plotting ---

# Approach 1: Combined model-language hue
df['model_language'] = df['model'] + " - " + df['language']

plt.figure(figsize=(14, 7))
sns.barplot(data=df, x="revision", y="accuracy", hue="model_language", palette="Set2", ci=None)
plt.title("Model Accuracy by Checkpoint, Model, and Language")
plt.xlabel("Training Step (Revision)")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.legend(title="Model - Language")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Approach 2: Facet by language with model hue
sns.catplot(
    data=df,
    x="revision",
    y="accuracy",
    hue="model",
    col="language",
    kind="bar",
    height=5,
    aspect=1.3,
    palette="Set2",
    ci=None,
)
plt.subplots_adjust(top=0.85)
plt.suptitle("Model Accuracy by Checkpoint and Language")
plt.show()
