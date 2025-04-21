# Intensity Estimation of Poisson Point Processes using Score Matching and Neural Networks

## Overview
This project explores intensity estimation for Poisson Point Processes (PPPs) using score matching as a loss function. It implements both parametric and non-parametric models, leveraging neural networks to approximate the intensity function without requiring explicit computation of the normalizing constant. The primary focus is on assessing the effectiveness of score matching for PPPs and addressing challenges related to domain truncation and numerical stability.

## Motivation
Gibbs point processes provide a framework for modeling spatial and temporal processes where interactions between points play a crucial role. However, likelihood-based inference is challenging due to an intractable normalizing constant. The Poisson Point Process is a special case where interactions are absent, simplifying the structure but retaining the difficulty of normalizing constant computation. This project utilizes the score matching approach introduced by Hyvärinen to estimate the intensity function of a PPP without explicit likelihood computation, offering an efficient alternative to traditional methods.

## Features
- **Score Matching for Poisson Point Processes**: Avoids direct computation of the normalizing constant.
- **Parametric and Non-Parametric Models**: Evaluates performance across different modeling approaches.
- **Neural Networks as Function Approximators**: Uses fully connected neural networks for non-parametric estimation.
- **Gaussian Weighting for Regularization**: Mitigates domain truncation issues in non-parametric models.
- **Scalable Experimentation Framework**: Supports running multiple configurations through automated scripts.

## Implementation Details
### Model Development
- **Parametric Models**: Utilize known analytical density functions for estimation.
- **Non-Parametric Models**: Use neural networks to approximate intensity functions.
- **Score Matching**: Employed as the loss function to guide model training.
- **Gaussian Weighting**: Applied to address domain truncation issues.

### Challenges and Solutions
- **Domain Truncation**: Managed using Gaussian weighting, though effectiveness varies.
- **Unbounded Intensity Outputs**: Regularization techniques are explored to mitigate extreme values.
- **Boundary Sensitivity in Non-Parametric Models**: Requires careful handling to ensure model stability.

## Project Structure
```
project_root/
│── models_notebooks/      # Jupyter notebooks for parametric and non-parametric models
│── run_experiment_poisson.py  # Functions to run Poisson non-parametric models
│── poisson_model.py       # Implementation of Poisson model functions
│── run_all_experiments.py # Script for running different experiment configurations
│── README.md              # Project documentation
```

## Usage
### Prerequisites
Ensure you have Python installed along with the necessary dependencies:
```sh
pip install -r requirements.txt
```

### Running Experiments
To run experiments with different configurations:
```sh
python run_all_experiments.py
```
For running specific Poisson model experiments:
```sh
python run_experiment_poisson.py
```

## Future Directions
- Refining weighting mechanisms to better handle boundary effects.
- Exploring alternative regularization techniques to control intensity inflation.
- Investigating adaptive weighting strategies for improved generalization.
- Extending the model to broader classes of point processes beyond Poisson assumptions.
- Testing in higher-dimensional spaces to validate robustness and adaptability.

## References
- Hyvärinen, A. "Estimation of Non-Normalized Statistical Models by Score Matching."

---
For any questions or contributions, feel free to open an issue or submit a pull request!
