Multivariate Normal Distribution Visualization
A simple Python script demonstrating bivariate normal distribution generation and visualization using NumPy and Matplotlib.

📌 Overview
This project generates and visualizes random samples from a multivariate normal distribution. It serves as an educational tool for understanding probability distributions, covariance matrices, and data visualization in Python.

🎯 Features
Bivariate Normal Distribution: Generate 500 random samples from a 2D normal distribution

Customizable Parameters: Mean and covariance matrix can be easily modified

Data Visualization: Scatter plot visualization of the generated distribution

Statistical Analysis: Basic shape and structure analysis of the generated data

🛠️ Technologies Used
Python 3.x

NumPy: Numerical computing and random sampling

Matplotlib: Data visualization and plotting

Pandas: Data manipulation (imported but not used in current implementation)

📊 Mathematical Background
The script uses the multivariate normal distribution defined by:

Parameters:
Mean vector (μ): [0.0, 0.0] - Center of the distribution

Covariance matrix (Σ): [[1, 0], [0, 1]] - Shape and orientation of the distribution

Properties:
500 samples generated from the distribution

Independent variables (covariance = 0 between dimensions)

Unit variance along both dimensions

Zero correlation between X and Y axes

🚀 Quick Start
Prerequisites
bash
pip install numpy matplotlib pandas
Running the Script
python
python multivariate_normal_visualization.py
Code Structure
python
import numpy as np
import matplotlib.pyplot as plt

# Define distribution parameters
mean = np.array([0.0, 0.0])          # Center at origin
cov = np.array([[1, 0], [0, 1]])    # Identity covariance matrix

# Generate 500 samples
samples = np.random.multivariate_normal(mean, cov, 500)

# Display shape and first few samples
print(f"Shape: {samples.shape}")
print(f"First 10 samples:\n{samples[:10]}")

# Create scatter plot
plt.scatter(samples[:, 0], samples[:, 1])
plt.title("Bivariate Normal Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
📈 Output
Console Output:
text
Shape: (500, 2)
First few samples:
[[-0.3337 -0.5801]
 [-0.2292  2.3372]
 [ 0.3146 -1.5167]
 ...]
Visualization:
A scatter plot showing 500 points randomly distributed around the origin

Circular pattern due to independent X and Y dimensions

Points concentrated near the center with density decreasing outward

🎨 Customization Examples
1. Change Distribution Center
python
mean = np.array([5.0, 6.0])  # Shift distribution to (5, 6)
2. Add Correlation Between Variables
python
cov = np.array([[1.3, 0.2],  # Positive correlation
                [0.2, 1.1]])
3. Generate Different Sample Size
python
samples = np.random.multivariate_normal(mean, cov, 1000)  # 1000 samples
4. Multiple Distributions (Commented in code)
python
# Uncomment to add a second distribution
mean2 = np.array([5.0, 6.0])
cov2 = np.array([[1.3, 0.2], [0.2, 1.1]])
dist2 = np.random.multivariate_normal(mean2, cov2, 500)
plt.scatter(dist2[:, 0], dist2[:, 1], color='red', alpha=0.5)
📊 Statistical Insights
Current Configuration:
Mean: (0, 0) - Distribution centered at origin

Variance: 1 for both X and Y axes

Covariance: 0 - X and Y are independent

Shape: Circular distribution (isotropic)

Expected Properties:
68% of points within 1 unit from center

95% of points within 2 units from center

99.7% of points within 3 units from center

🔬 Educational Applications
Learning Concepts:
Multivariate Normal Distribution: Understanding 2D probability distributions

Covariance Matrix: How it controls shape and orientation

Statistical Independence: Zero covariance implies independent variables

Data Generation: Creating synthetic datasets for testing algorithms

Extensions for Learning:
python
# Calculate empirical statistics
empirical_mean = np.mean(samples, axis=0)
empirical_cov = np.cov(samples.T)

print(f"Empirical Mean: {empirical_mean}")
print(f"Empirical Covariance:\n{empirical_cov}")
📁 Project Structure
text
multivariate-normal/
├── multivariate_normal.py    # Main script
├── README.md                 # This documentation
├── requirements.txt          # Dependencies
└── examples/                # Additional examples
    ├── correlated_dist.py   # Correlated distribution example
    └── multiple_dists.py    # Multiple distributions example
📋 Requirements
Create requirements.txt:

txt
numpy>=1.19.0
matplotlib>=3.3.0
pandas>=1.1.0
🎓 Learning Resources
Related Concepts:
Probability Density Function (PDF) of multivariate normal

Mahalanobis distance for multivariate outliers

Principal Component Analysis (PCA) using eigen decomposition

Gaussian Mixture Models (GMM) for clustering

Next Steps:
Add probability density contour plots

Implement outlier detection using Mahalanobis distance

Create interactive visualizations with Plotly

Extend to 3D distributions

🐛 Common Issues & Solutions
Issue 1: Singular Covariance Matrix
python
# Solution: Ensure covariance matrix is positive definite
cov = np.array([[1, 0.5], [0.5, 1]])  # Valid
# cov = np.array([[1, 1], [1, 1]])    # Invalid (singular)
Issue 2: Memory Error with Large Samples
python
# Solution: Generate in batches or reduce sample size
samples = np.random.multivariate_normal(mean, cov, 10000)  # May be large
Issue 3: Non-positive Definite Matrix
python
# Solution: Check eigenvalues
eigenvalues = np.linalg.eigvals(cov)
if np.any(eigenvalues <= 0):
    print("Covariance matrix not positive definite!")
🤝 Contributing
Feel free to:

Fork the repository

Add more visualization options

Implement statistical tests

Add interactive features

Create educational examples

📄 License
This project is open source and available under the MIT License.

🙏 Acknowledgments
NumPy documentation for random.multivariate_normal

Matplotlib for visualization capabilities

Statistical learning resources that inspired this example

