# Adversarial Generation of Banknotes

Modern approaches to detecting counterfeit currency are increasingly relying upon Convolutional Neural Network (CNN) based image detection tools. However, these systems are susceptible to high-fidelity counterfeits generated through automated machine learning. A sophisticated attacker can automate the generation of counterfeit currencies designed to circumvent specific static detection models, allowing them to rapidly adapt their counterfeits as detection systems are updated. This vulnerability creates a gap between the bill the CNNs are trained on, and the actual bills they see in deployment. This increased false negative rate exposes legitimate vendors and banks to significant financial risk.  

This project investigates the vulnerabilities of static Convolutional Neural Network (CNN) counterfeit currency detection systems. By automating the generation of high-fidelity counterfeit Bangladeshi banknotes (NoteShieldBD/JaalTaka dataset) using Generative Adversarial Networks (GANs), we demonstrate the need for more robust, dynamically updated detection mechanisms.

The project compares two GAN architectures to fool a target multi-view CNN:
1.  **Architecture A (Baseline):** A Multi-Layer Perceptron (MLP) GAN.
2.  **Architecture B (Advanced):** A Multi-View Convolutional GAN (DCGAN Variant) with specialized branches for each of the 6 banknote segments.

We evaluate the success of these attacks using Adversarial Success Rate (Fooling Rate), Fréchet Inception Distance (FID), and Grad-CAM heatmaps.

## Project Structure

*   `data/`: Contains raw and processed datasets (expected as 6-segment tensors).
*   `src/`: Core source code (data loading, models, training loops, evaluation metrics).
*   `scripts/`: Executable scripts for training and evaluation.
*   `notebooks/`: Jupyter notebooks for EDA, prototyping, and visualization.

## Development Setup

1.  **Create a conda environment:**
    ```bash
    conda create -n cpen355 python=3.10
    conda activate cpen355
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1.  **Prepare the Data:**
    Place the NoteShieldBD (JaalTaka) dataset into `data/raw/` and ensure it is preprocessed into the expected 6-segment tensor format `(6, H, W, C)` normalized to `[-1, 1]`.

2.  **Train the Baseline (MLP GAN):**
    ```bash
    python scripts/train_mlp.py
    ```

3.  **Train the Advanced Model (DCGAN Variant):**
    ```bash
    python scripts/train_dcgan.py
    ```

4.  **Evaluate Models:**
    ```bash
    python scripts/evaluate.py
    ```
    This script will calculate the Fooling Rate against the target model and the FID scores. Use the notebooks for generating Grad-CAM heatmaps.
