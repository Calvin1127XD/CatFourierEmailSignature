import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg

# Load the contour data (assuming the contour points are already extracted)
contour = np.load('cat_iOS_contour.npy')

# Convert the contour points to complex numbers
contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]

# Normalize and scale the contour to fit the canvas (flip the imaginary part to correct orientation)
contour_complex -= np.mean(contour_complex)
contour_complex = contour_complex.real - 1j * contour_complex.imag  # Flip the imaginary component
scale_factor = 500 / max(np.abs(contour_complex))  # Fit within [-500, 500]
contour_complex *= scale_factor

# Number of points in the contour
N = len(contour_complex)

# Define Fourier coefficients using vectorized method
def calculate_fourier_coeffs(contour, num_coeffs):
    coeffs = []
    for n in range(-num_coeffs, num_coeffs + 1):
        integrand = contour * np.exp(-2j * np.pi * n * np.arange(N) / N)
        coeff = np.sum(integrand) / N
        coeffs.append((n, coeff))
    return sorted(coeffs, key=lambda x: np.abs(x[1]), reverse=True)

# Get a limited number of Fourier coefficients (you can adjust this)
num_components = 35  # Adjust to control the number of components
fourier_coeffs = calculate_fourier_coeffs(contour_complex, num_components)

# Set up the figure for the animation and image subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14), gridspec_kw={'height_ratios': [3, 1.3]})

# Adjust the spacing between the two subplots (reduce gap)
plt.subplots_adjust(hspace=0.05, top=0.9, bottom=0.1)  # Decrease hspace to almost zero

# Remove axes and title for email signature
ax1.set_xlim(-500, 500)
ax1.set_ylim(-500, 500)
ax1.set_aspect('equal')
ax1.axis('off')  # Remove axes
ax2.axis('off')  # Remove axes from image subplot

# Load and display the original cat image in the lower subplot
cat_image = mpimg.imread('cat_iOS_original.jpg')
ax2.imshow(cat_image)

# Initialize the line for the traced path
trace, = ax1.plot([], [], 'r-', lw=2)

# Initialize lists for arrows and epicycles
arrows = []
epicycles = []

# Add arrow objects and epicycles for each component
for i in range(len(fourier_coeffs)):
    # Calculate arrow length for proportional head size
    arrow_length = np.abs(fourier_coeffs[i][1]) / 80
    # Create an arrow with filled triangle heads (adjust mutation_scale proportional to length)
    arrow = FancyArrowPatch((0, 0), (0, 0), color='blue', arrowstyle='-|>', mutation_scale=5 * arrow_length, lw=1.5)
    ax1.add_patch(arrow)
    arrows.append(arrow)
    
    # Epicycles for visualization
    epicycle, = ax1.plot([], [], 'grey', lw=0.5, linestyle='--')
    epicycles.append(epicycle)

# Store the path traced by the tip of the last arrow
path = []

# Function to initialize the animation
def init():
    trace.set_data([], [])
    for arrow in arrows:
        arrow.set_positions((0, 0), (0, 0))
    for epicycle in epicycles:
        epicycle.set_data([], [])
    return trace, *arrows, *epicycles

# Function to animate each frame
def animate(t):
    global path
    z = 0
    arrow_positions = [z]

    # Calculate position of arrow heads for each coefficient
    for n, coeff in fourier_coeffs:
        z += coeff * np.exp(2j * np.pi * n * t / N)
        arrow_positions.append(z)

    # Update arrows and epicycles
    for i in range(len(arrow_positions) - 1):
        # Update arrow
        start = (arrow_positions[i].real, arrow_positions[i].imag)
        # Slightly extend the end of the arrow to reduce gaps
        end = (arrow_positions[i + 1].real, arrow_positions[i + 1].imag)
        arrows[i].set_positions(start, end)

        # Update the epicycle (grey circle)
        radius = np.abs(fourier_coeffs[i][1])
        theta = np.linspace(0, 2 * np.pi, 100)
        epicycles[i].set_data(
            arrow_positions[i].real + radius * np.cos(theta),
            arrow_positions[i].imag + radius * np.sin(theta)
        )

    # Update the path traced by the final arrow
    path.append(arrow_positions[-1])
    trace.set_data([p.real for p in path], [p.imag for p in path])

    return trace, *arrows, *epicycles

# Create the animation
anim = FuncAnimation(fig, animate, frames=np.linspace(0, N, 180), init_func=init,
                     interval=5, blit=True)

# Save the animation as a GIF
anim.save('cat_fourier.gif', writer='imagemagick', fps=15)

# Show the animation (optional if you're just saving)
# plt.show()
