import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def calculate_cutoff_frequency(n, m, a, b, ur, epsr):
    """Calculate the cutoff frequency for the given mode."""
    c = 3e8  # speed of light in vacuum
    fc = (c / (2 * np.pi)) * np.sqrt(
        ((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2) / (ur * epsr)
    )
    return np.real(fc)  # Return real part for comparison


def get_char_vals(ur, epsr, m, n, f, a, b):
    mu0 = 4 * np.pi * 1e-7
    eps0 = 8.85e-12
    omega = 2 * np.pi * f
    # Wave numbers
    k = omega * np.sqrt(mu0 * ur * eps0 * epsr)
    kx = m * np.pi / a
    ky = n * np.pi / b
    h = np.sqrt(kx**2 + ky**2)
    kz = gamma = np.sqrt(k**2 - h**2)
    return k, kx, ky, omega, kz, gamma, h


def calculate_fields(x, y, z, f, ur, epsr, n, m, mode, a, b):
    """Calculate the electromagnetic field components for TE/TM modes."""
    mu0 = 4 * np.pi * 1e-7
    eps0 = 8.85e-12
    k, kx, ky, omega, kz, gamma, h = get_char_vals(ur, epsr, m, n, f, a, b)

    # Mode amplitude factors
    Ex = Ey = Ez = Hx = Hy = Hz = np.zeros_like(x, dtype=complex)

    beta = np.real(np.sqrt(k**2 - kx**2 - ky**2))
    alpha = -np.imag(np.sqrt(k**2 - kx**2 - ky**2))

    if mode == "TE":
        # TE mode field components
        Hz = np.cos(kx * x) * np.cos(ky * y) * np.exp(-gamma * z)
        Hx = (
            (1j * beta * kx / h**2)
            * np.sin(kx * x)
            * np.cos(ky * y)
            * np.exp(-(alpha + 1j * beta) * z)
        )
        Hy = (
            (1j * beta * ky / h**2)
            * np.cos(kx * x)
            * np.sin(ky * y)
            * np.exp(-(alpha + 1j * beta) * z)
        )
        Ex = (
            (1j * omega * mu0 * ur * ky / h**2)
            * np.cos(kx * x)
            * np.exp(-(alpha + 1j * beta) * z)
        )
        Ey = (
            (-1j * omega * mu0 * ur * kx / h**2)
            * np.sin(kx * x)
            * np.exp(-(alpha + 1j * beta) * z)
        )

    elif mode == "TM":
        # TM mode field components
        Ez = np.sin(kx * x) * np.sin(ky * y) * np.exp(-gamma * z)
        Ex = (
            (-1j * beta * kx / h**2)
            * np.cos(kx * x)
            * np.sin(ky * y)
            * np.exp(-(alpha + 1j * beta) * z)
        )
        Ey = (
            (-1j * beta * ky / h**2)
            * np.sin(kx * x)
            * np.cos(ky * y)
            * np.exp(-(alpha + 1j * beta) * z)
        )
        Hx = (
            (1j * omega * eps0 * epsr * ky / h**2)
            * np.sin(kx * x)
            * np.cos(ky * y)
            * np.exp(-(alpha + 1j * beta) * z)
        )
        Hy = (
            (-1j * omega * eps0 * epsr * kx / h**2)
            * np.cos(kx * x)
            * np.sin(ky * y)
            * np.exp(-(alpha + 1j * beta) * z)
        )

    return Ex, Ey, Ez, Hx, Hy, Hz


def plot_field_components(X, Y, Ex, Ey, Hx, Hy, mode, n, m, z, plane="z"):
    """Create plots for field components in specified plane."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot E-field components
    ax1.quiver(X, Y, np.real(Ex), np.real(Ey), color="r")
    ax1.set_title(f"E-field Vector Pattern\n{mode}{n}{m} at {plane}={z:.3f}m")
    ax1.set_xlabel("x (m)" if plane != "x" else "y (m)")
    ax1.set_ylabel("y (m)" if plane != "y" else "z (m)")
    ax1.set_aspect("equal")
    ax1.grid(True)

    # Plot E-field magnitude
    E_mag = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
    contour_e = ax3.contourf(X, Y, E_mag)
    plt.colorbar(contour_e, ax=ax3)
    ax3.set_title("E-field Magnitude")
    ax3.set_xlabel("x (m)" if plane != "x" else "y (m)")
    ax3.set_ylabel("y (m)" if plane != "y" else "z (m)")
    ax3.set_aspect("equal")

    # Plot H-field components
    ax2.quiver(X, Y, np.real(Hx), np.real(Hy), color="b")
    ax2.set_title(f"H-field Vector Pattern\n{mode}{n}{m} at {plane}={z:.3f}m")
    ax2.set_xlabel("x (m)" if plane != "x" else "y (m)")
    ax2.set_ylabel("y (m)" if plane != "y" else "z (m)")
    ax2.set_aspect("equal")
    ax2.grid(True)

    # Plot H-field magnitude
    H_mag = np.sqrt(np.abs(Hx) ** 2 + np.abs(Hy) ** 2)
    contour_h = ax4.contourf(X, Y, H_mag)
    plt.colorbar(contour_h, ax=ax4)
    ax4.set_title("H-field Magnitude")
    ax4.set_xlabel("x (m)" if plane != "x" else "y (m)")
    ax4.set_ylabel("y (m)" if plane != "y" else "z (m)")
    ax4.set_aspect("equal")

    plt.tight_layout()
    return fig


def main():
    # Set page config
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

    # Set matplotlib style for light mode
    plt.style.use("default")

    st.title("Rectangular Waveguide Field Visualization")

    # Sidebar inputs
    with st.sidebar:
        st.header("Input Parameters")

        # Operating parameters
        st.subheader("Operating Parameters")
        f = st.number_input(
            "Frequency (Hz)", min_value=1e6, max_value=1e12, value=1e9, format="%e"
        )
        mode = st.selectbox("Mode", ["TM", "TE"])

        # Material parameters
        st.subheader("Material Parameters")
        ur_real = st.number_input("μᵣ (Real part)", value=1.0, format="%.4f")
        ur_imag = st.number_input("μᵣ (Imaginary part)", value=0.5, format="%.4f")
        ur = ur_real - 1j * ur_imag

        epsr_real = st.number_input("εᵣ (Real part)", value=1.0, format="%.4f")
        epsr_imag = st.number_input("εᵣ (Imaginary part)", value=0.5, format="%.4f")
        epsr = epsr_real - 1j * epsr_imag

        # Geometry parameters
        st.subheader("Geometry Parameters")
        a = st.number_input("Waveguide width a (m)", min_value=0.001, value=1.0)
        b = st.number_input("Waveguide height b (m)", min_value=0.001, value=1.0)

        # Mode numbers
        st.subheader("Mode Numbers")
        try:
            n = st.number_input("Mode number n", min_value=0, max_value=10, value=1)
            m = st.number_input("Mode number m", min_value=0, max_value=10, value=1)
            if mode == "TE" and (n == 0 and m == 0):
                st.error("Error: TE mode requires cannot have both n and m as 0")
                st.stop()
        except ValueError:
            st.error("Error: Please enter valid mode numbers")

    # Calculate and display important parameters at the top
    fc = calculate_cutoff_frequency(n, m, a, b, ur, epsr)
    k, kx, ky, omega, kz, gamma, h = get_char_vals(ur, epsr, m, n, f, a, b)
    print(ur, epsr, gamma)
    beta = np.real(np.sqrt(k**2 - kx**2 - ky**2))
    alpha = -np.imag(np.sqrt(k**2 - kx**2 - ky**2))

    # Display key parameters in columns at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cutoff Frequency (Hz)", f"{fc:.2e}")
    with col2:
        st.metric("Phase Constant (rad/m)", f"{beta:.4f}")
    with col3:
        st.metric("Attenuation Constant (Np/m)", f"{alpha:.4f}")
    with col4:
        wavelength_z = 2 * np.pi / beta if beta > 0 else float("inf")
        st.metric("Guide Wavelength (m)", f"{wavelength_z:.4f}")

    # Check if frequency is above cutoff
    if f <= fc:
        st.error(
            f"Error: Operating frequency ({f:.2e} Hz) must be above cutoff frequency ({fc:.2e} Hz)"
        )
        st.stop()

    # Cross-section selection
    plane = st.radio("Select Cross-section Plane", ["z", "x", "y"], horizontal=True)

    # Create appropriate mesh grid based on selected plane
    if plane == "z":
        z = st.slider("Z position (m)", 0.0, wavelength_z, value=wavelength_z / 2)
        x = np.linspace(0, a, 15)
        y = np.linspace(0, b, 15)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z)
    elif plane == "x":
        x = st.slider("X position (m)", 0.0, a, value=a / 2)
        y = np.linspace(0, b, 15)
        z = np.linspace(0, wavelength_z, 15)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, x)
    else:  # plane == 'y'
        y = st.slider("Y position (m)", 0.0, b, value=b / 2)
        x = np.linspace(0, a, 15)
        z = np.linspace(0, wavelength_z, 15)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, y)

    # Calculate fields
    Ex, Ey, Ez, Hx, Hy, Hz = calculate_fields(X, Y, Z, f, ur, epsr, n, m, mode, a, b)

    # Plot appropriate field components based on plane
    if plane == "z":
        fig = plot_field_components(X, Y, Ex, Ey, Hx, Hy, mode, n, m, z, plane)
    elif plane == "x":
        fig = plot_field_components(Y, Z, Ey, Ez, Hy, Hz, mode, n, m, x, plane)
    else:  # plane == 'y'
        fig = plot_field_components(X, Z, Ex, Ez, Hx, Hz, mode, n, m, y, plane)

    # Display plot
    st.pyplot(fig)


if __name__ == "__main__":
    main()
