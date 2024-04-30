
import os
import trimesh

def rescale_mesh(mesh):
    """Rescales a mesh to 5 cm."""
    current_scale = mesh.scale  # Assuming 'scale' refers to the mesh size in some unit
    scale_factor = 5.0 / current_scale
    mesh.apply_scale(scale_factor)
    return mesh


def rescale_stl_files(folder_path):
  """
  Iterates through all subdirectories in a folder, resizes .stl files to 5 cm, and saves them with the same name.

  Args:
      folder_path (str): The path to the folder containing subdirectories with STL files.
  """
  for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
      if filename.lower().endswith('.stl'):
        filepath = os.path.join(dirpath, filename)
        # Load the mesh
        mesh = trimesh.load(filepath)
        # Rescale the mesh to 5 cm (your preference)
        rescaled_mesh = rescale_mesh(mesh.copy())
        # Save the rescaled mesh
        rescaled_mesh.export(filepath)
        print(f"Rescaled '{filepath}' to 5 cm.")


# Example usage
folder_to_process = "D:\ShapeNet\shapenet-watertight"
rescale_stl_files(folder_to_process)
