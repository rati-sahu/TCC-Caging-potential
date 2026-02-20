# This code calculates the Number of particles within a sphere of radius r as a function of r. 
#This gives the fractal dimension that tells about the spatial organisation of the particles

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import linregress
from scipy.io import savemat

plt.rcParams.update({'font.size': 22})


sigma = 1.6
max_origins = 100 

pathXYZ = 'E:/p3_TCC/tcc/xyzLinked/dec/all/10B/d1/'
coords = np.loadtxt(pathXYZ + 'xyz_10b_2-12-1_t=00.txt')

N = coords.shape[0]
print(f"Total particles: {N}")

xmin, ymin, zmin = coords.min(axis=0)
xmax, ymax, zmax = coords.max(axis=0)

box_dims = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
max_possible_r = np.min(box_dims) / 2.0

print(f"Min Box Dimension: {np.min(box_dims):.2f}")
print(f"Max theoretical r: {max_possible_r:.2f}")

r_values = np.linspace(1.5 * sigma, min(7, max_possible_r*0.95), 20)

tree = cKDTree(coords)
rng = np.random.default_rng()

Nr_mean = np.zeros(len(r_values))
Nr_std  = np.zeros(len(r_values))
n_used  = np.zeros(len(r_values), dtype=int)

print("Calculating N(r)...")

# Compute N(r)

for i, r in enumerate(r_values):

    bulk_mask = (
        (coords[:, 0] - xmin >= r) & (xmax - coords[:, 0] >= r) &
        (coords[:, 1] - ymin >= r) & (ymax - coords[:, 1] >= r) &
        (coords[:, 2] - zmin >= r) & (zmax - coords[:, 2] >= r)
    )

    valid_indices = np.where(bulk_mask)[0]

    if len(valid_indices) == 0:
        Nr_mean[i] = np.nan
        continue

    if max_origins == -1 or len(valid_indices) <= max_origins:
        origin_indices = valid_indices
    else:
        origin_indices = rng.choice(valid_indices, max_origins, replace=False)

    n_used[i] = len(origin_indices)

    origin_coords = coords[origin_indices]
    neighbors_list = tree.query_ball_point(origin_coords, r)

    counts = [len(n) for n in neighbors_list]

    Nr_mean[i] = np.mean(counts)
    Nr_std[i]  = np.std(counts)


# Fit Fractal Dimension (log–log slope)

valid_mask = ~np.isnan(Nr_mean) & (Nr_mean > 0)
r_clean = r_values[valid_mask]
N_clean = Nr_mean[valid_mask]

log_r = np.log(r_clean)
log_N = np.log(N_clean)

slope, intercept, r_value, p_value, std_err = linregress(log_r, log_N)
D_f = slope

print(f"Estimated Fractal Dimension D_f: {D_f:.3f}")

# Prepare Fit and Reference Lines


fit_curve = np.exp(intercept) * r_clean**slope

if len(r_clean) > 0:
    r0 = r_clean[0]
    N0 = fit_curve[0]

    slope3_line = N0 * (r_clean / r0)**3
    slope2_line = N0 * (r_clean / r0)**2
else:
    slope3_line = np.array([])
    slope2_line = np.array([])


# Save Data for MATLAB


save_txt_path = "E:/p3_TCC/tcc/codes_new/further_analysis/data/Nr/10b_d1_2-7_nr_data.txt"
save_mat_path = "E:/p3_TCC/tcc/codes_new/further_analysis/data/Nr/10b_d1_2-7_nr_data.mat"

# Save MATLAB file
savemat(
    save_mat_path,
    {
        "r": r_clean,
        "N": N_clean,
        "fit": fit_curve,
        "slope3": slope3_line,
        "slope2": slope2_line,
        "D_f": D_f,
        "intercept": intercept,
        "std_err": std_err
    }
)

# Save text file
data_to_save = np.column_stack((
    r_clean,
    N_clean,
    fit_curve,
    slope3_line,
    slope2_line
))

np.savetxt(
    save_txt_path,
    data_to_save,
    header=f"""r    N(r)    fit    slope3    slope2
# Fractal dimension D_f = {D_f:.6f}
# Intercept = {intercept:.6f}
# Std error = {std_err:.6f}
""",
    comments=''
)

print("Data successfully saved (.txt and .mat).")

# Plot

plt.figure(figsize=(7, 6))

plt.loglog(r_clean, N_clean, 'o', alpha=0.6, label="N(r)")
plt.loglog(r_clean, fit_curve, 'r-', lw=2, label=f"$d_f = {D_f:.2f}$")

plt.loglog(r_clean, slope3_line, 'k--', alpha=0.3, label="Slope 3")
plt.loglog(r_clean, slope2_line, 'g--', alpha=0.3, label="Slope 2")

plt.xlabel(r"$r$")
plt.ylabel(r"$N(r)$")
plt.title("10B")
plt.legend(fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.show()
