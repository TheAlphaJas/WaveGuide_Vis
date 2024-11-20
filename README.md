# Rectangular Waveguide Field Visualization

This project provides an interactive simulation for visualizing electromagnetic field distributions in a **rectangular waveguide** for both **TE (Transverse Electric)** and **TM (Transverse Magnetic)** modes. The simulation is built using **Streamlit**, **NumPy**, and **Matplotlib**, and includes the following features:

- **Complex permittivity (\( \epsilon_r \)) and permeability (\( \mu_r \))** for modeling attenuation.
- Visualization of fields across different cross-sections (\( xy \), \( yz \), and \( zx \) planes).
- Calculation of key waveguide parameters like cutoff frequency, phase constant, attenuation constant, and guide wavelength.
- **Dynamic field plotting** with user-adjustable parameters such as frequency, mode, geometry, and material properties.

---

## Features

### 1. **Cutoff Frequency Calculation**
   - Computes the cutoff frequency for a given mode and waveguide geometry.
   - Supports both TE and TM modes.

### 2. **Electromagnetic Field Visualization**
   - Generates field distributions for \( E_x, E_y, E_z \) and \( H_x, H_y, H_z \).
   - Visualizes the fields on cross-sections: \( xy \), \( yz \), and \( zx \).

### 3. **Waveguide Parameter Computations**
   - **Phase Constant (\( \beta \))**
   - **Attenuation Constant (\( \alpha \))** (for lossy materials)
   - **Guide Wavelength (\( \lambda_g \))**

### 4. **Interactive Interface**
   - Adjustable input parameters:
     - **Frequency:** Select operational frequency for visualization.
     - **Mode:** Specify TE or TM modes and indices \( m \), \( n \).
     - **Waveguide Dimensions:** Set width and height.
     - **Material Properties:** Customize \( \epsilon_r \) and \( \mu_r \).

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Streamlit**
- **NumPy**
- **Matplotlib**

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/waveguide-visualization.git
   cd waveguide-visualization
   ```
2. Install requirements and run file:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

This repo was created as a course project for EE305 - EM Waves (Autumn 2024), @ Indian Institute of Technology Indore, India

Credits to our course instructor [Dr. Saptarshi Ghosh](http://people.iiti.ac.in/~sghosh/)
