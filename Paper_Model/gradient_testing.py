import json
import matplotlib.pyplot as plt

# Load the saved data file from the main folder above data/route
file_path = r"data\route\2026 Sasol Solar Challenge Route (Publish)_Day 1 _10 Sept Stage 2 Rustenburg to Swartruggens.kml.save"  # Adjust filename if needed

with open(file_path, "r") as f:
    data = json.load(f)

# Extract distance and gradient arrays from the profile
distances = data["profile"]["Distance"]
gradients = data["profile"]["Gradient"]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(distances, gradients, color='red', linewidth=1.5, label='Gradient (%)')

# Styling the plot
plt.title('Route Gradient vs. Distance', fontsize=14, fontweight='bold')
plt.xlabel('Cumulative Distance (km)', fontsize=12)
plt.ylabel('Gradient (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.legend()

plt.tight_layout()
plt.show()