import plyfile

# Read the original PLY file
plydata = plyfile.PlyData.read('PLYFILE_PATH')  # Update this to your actual PLY file path
vertex = plydata['vertex']

# To avoid the "ValueError: cannot set an array element with a sequence" error, 
# Need to create a new structured array with the updated dtype names instead of trying to modify the dtype of the existing array in place.
old_names = vertex.data.dtype.names
new_names = []

for name in old_names:
    # x, y, z, nx, ny, nz를 제외한 b1, b2... 이름 앞에 scalar_ 추가
    if name.startswith('b') and not name.startswith('scalar_'):
        new_names.append('scalar_' + name)
    else:
        new_names.append(name)

# Change the dtype names of the vertex data (this creates a new structured array with the new names)
vertex.data.dtype.names = new_names

# Create a new PlyElement with the updated vertex data and save it to a new PLY file
new_vertex = plyfile.PlyElement.describe(vertex.data, 'vertex')
new_ply = plyfile.PlyData([new_vertex], text=False) # Binary 저장

# Save the new PLY file
new_ply.write('PLYFILE_OUTPUT_PATH')  # Update this to your desired output PLY file path
print("Completed renaming scalar fields and saved to new PLY file. Open the new PLY file in CloudCompare to verify that the scalar fields are now visible.")