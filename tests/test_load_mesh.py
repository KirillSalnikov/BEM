"""Test load_mesh with OBJ and STL formats."""
import numpy as np
from bem_core import load_mesh, build_rwg
import tempfile, os

# Create a simple cube OBJ
cube_obj = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

with tempfile.NamedTemporaryFile(suffix='.obj', mode='w', delete=False) as f:
    f.write(cube_obj)
    obj_path = f.name

try:
    verts, tris = load_mesh(obj_path)
    rwg = build_rwg(verts, tris)
    print(f"OBJ cube: {len(verts)} verts, {len(tris)} tris, {rwg['N']} RWG")
    assert len(verts) == 8
    assert len(tris) == 12
    # Closed surface: 12 tris, 18 edges (all interior for closed cube)
    assert rwg['N'] == 18, f"Expected 18 RWG, got {rwg['N']}"
    print("  OK: closed surface, all edges interior")
finally:
    os.unlink(obj_path)

# Test STL (ASCII)
stl_ascii = """\
solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 1 1 0
      vertex 0 1 0
      vertex 0 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 1 1 0
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 1 0
      vertex 0 0 0
      vertex 0 0 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 1 0
      vertex 0 0 1
      vertex 0 1 1
    endloop
  endfacet
endsolid cube
"""

with tempfile.NamedTemporaryFile(suffix='.stl', mode='w', delete=False) as f:
    f.write(stl_ascii)
    stl_path = f.name

try:
    verts2, tris2 = load_mesh(stl_path)
    rwg2 = build_rwg(verts2, tris2)
    print(f"STL cube: {len(verts2)} verts, {len(tris2)} tris, {rwg2['N']} RWG")
    assert len(verts2) == 8
    assert len(tris2) == 12
    assert rwg2['N'] == 18
    print("  OK: closed surface")
finally:
    os.unlink(stl_path)

print("\nAll mesh loading tests passed!")
