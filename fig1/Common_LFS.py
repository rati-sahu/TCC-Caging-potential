# This code makes the .xyz file to indetify the common particles between FCC and other clusters.

import numpy as np

path = "E:/p3_TCC/tcc/codes_new/tcc_shear/cluster_xyzs/cry/"
fcc_xyz  = path + "feat_crystal_sio2_1.5um_day2_9911_779.xyz.FCC_clusts.xyz"
f11_xyz  = path + "feat_crystal_sio2_1.5um_day2_9911_779.xyz.11F_clusts.xyz"
out_xyz  = path + "crystal_11F_12E_joint.xyz"

tol = 1e-6   # coordinate matching tolerance

def read_xyz_frames(filename):
    """Read xyz file into list of frames.
       Each frame is (comment, coords[N,3])"""
    frames = []

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            N = int(line.strip())
            comment = f.readline().strip()

            coords = np.zeros((N, 3))
            for i in range(N):
                parts = f.readline().split()
                coords[i] = list(map(float, parts[1:4]))

            frames.append((comment, coords))

    return frames


def coords_to_set(coords, tol):
    """Convert coordinates to a set of rounded tuples for matching."""
    return set(tuple(np.round(c / tol).astype(int)) for c in coords)


# Read xyz trajectories

fcc_frames = read_xyz_frames(fcc_xyz)
f11_frames = read_xyz_frames(f11_xyz)

assert len(fcc_frames) == len(f11_frames), "Number of frames must match"

# Initialize overlap counters
total_common = 0
total_11f = 0

with open(out_xyz, "w") as fout:

    for frame_id, ((cmt_fcc, fcc_coords), (cmt_11f, f11_coords)) in enumerate(
        zip(fcc_frames, f11_frames)
    ):

        fcc_set = coords_to_set(fcc_coords, tol)
        f11_set = coords_to_set(f11_coords, tol)

        # ---- Overlap calculation ----
        common = fcc_set.intersection(f11_set)

        n_11f = len(f11_set)
        n_common = len(common)

        if n_11f > 0:
            frac = n_common / n_11f
        else:
            frac = 0.0

        print(f"Frame {frame_id}: fraction of 11F also in 12E = {frac:.3f}")

        # accumulate totals
        total_common += n_common
        total_11f += n_11f

        # ---- Combine for writing ----
        all_coords = np.vstack((fcc_coords, f11_coords))
        all_keys = [tuple(np.round(c / tol).astype(int)) for c in all_coords]

        fout.write(f"{len(all_coords)}\n")
        fout.write(f"Frame {frame_id}\n")

        for coord, key in zip(all_coords, all_keys):

            in_fcc = key in fcc_set
            in_11f = key in f11_set

            if in_fcc and not in_11f:
                ptype = 1
            elif in_fcc and in_11f:
                ptype = 2
            elif in_11f and not in_fcc:
                ptype = 3
            else:
                continue

            fout.write(f"{ptype} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


# Overall fraction

if total_11f > 0:
    overall_fraction = total_common / total_11f
    print(f"Overall fraction of 11F also in 12E = {overall_fraction:.3f}")
    
print(f"\nCombined XYZ written to: {out_xyz}")
