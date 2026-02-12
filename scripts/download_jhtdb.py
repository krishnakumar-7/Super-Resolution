import os
import time
import zeep
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
DATASET_NAME = "isotropic1024coarse"
NUM_SNAPSHOTS = 1000
START_TIME = 0.0
END_TIME = 10.0
GRID_SIZE = 64
Z_SLICE_LOCATION = 0.0
PLOT_INTERVAL = 10
BACKUP_INTERVAL = 100
JHTDB_URL = 'http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL'
AUTH_TOKEN = "edu.jhu.pha.turbulence.testing-201302"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_visualization(velocity_field, t, step_idx):
    """Generates and saves a 3-panel plot of velocity components."""
    u = velocity_field[:, :, 0]
    v = velocity_field[:, :, 1]
    w = velocity_field[:, :, 2]

    # Determine common color scale
    v_min = min(u.min(), v.min(), w.min())
    v_max = max(u.max(), v.max(), w.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = [('u', u), ('v', v), ('w', w)]

    for ax, (name, data) in zip(axes, components):
        im = ax.imshow(data, origin='lower', cmap='jet',
                       vmin=v_min, vmax=v_max, extent=[0, 6.28, 0, 6.28])
        ax.set_title(f"{name}-velocity (t={t:.3f})")
        ax.set_xlabel("x (rad)")
        ax.set_ylabel("y (rad)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"components_t_{step_idx:04d}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():

    print("--- Connecting to JHTDB Server ---")
    try:
        client = zeep.Client(JHTDB_URL)
        Point3 = client.get_type('ns0:Point3')
        ArrayOfPoint3 = client.get_type('ns0:ArrayOfPoint3')
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Define Spatial Grid
    x_range = np.linspace(0, 2 * np.pi, GRID_SIZE)
    y_range = np.linspace(0, 2 * np.pi, GRID_SIZE)

    point_list = [
        Point3(x=float(x), y=float(y), z=float(Z_SLICE_LOCATION))
        for y in y_range for x in x_range
    ]
    points_array = ArrayOfPoint3(Point3=point_list)

    # Time Snapshots
    time_snapshots = np.linspace(START_TIME, END_TIME, NUM_SNAPSHOTS)
    collected_data = []

    print(f"--- Starting Download ({NUM_SNAPSHOTS} snapshots) ---")

    for i, t in enumerate(time_snapshots):
        print(f"[{i+1}/{NUM_SNAPSHOTS}] Requesting t={t:.4f}...",
              end=" ", flush=True)

        try:
            response = client.service.GetVelocity(
                authToken=AUTH_TOKEN,
                dataset=DATASET_NAME,
                time=float(t),
                spatialInterpolation="Lag4",
                temporalInterpolation="None",
                points=points_array
            )

            # Parse response
            velocity_flat = np.array([[p.x, p.y, p.z] for p in response])
            velocity_field = velocity_flat.reshape(GRID_SIZE, GRID_SIZE, 3)
            collected_data.append(velocity_field)

            print("Done.", end=" ")

            # Visualization
            if i % PLOT_INTERVAL == 0:
                save_visualization(velocity_field, t, i)
                print("(Plot Saved)", end=" ")

            # Periodic Backup
            if (i + 1) % BACKUP_INTERVAL == 0:
                backup_path = os.path.join(
                    OUTPUT_DIR, f"backup_step_{i+1}.npy")
                np.save(backup_path, np.array(collected_data))

            print("")

        except Exception as e:
            print(f"\nError at t={t}: {e}")
            time.sleep(1)
            continue

    # Final Save
    final_array = np.array(collected_data)
    save_path = os.path.join(OUTPUT_DIR, "unlimited_coarse_data.npy")
    np.save(save_path, final_array)
    print(f"\n--- Download Complete. Shape: {final_array.shape} ---")


if __name__ == "__main__":
    main()
