# src/neuthos/geometry.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

@dataclass
class Geometry:
    # --- geometry inputs ---
    naxial: int            # number of axial nodes (-)
    nass: int              # number of assemblies in core row/column (-)
    npin: int              # number of pins in assembly row/column (-)
    latsym: str            # lattice symmetry: 'full', 'se'
    ngtubes: int           # number of guide tubes in assembly (-)
    nitubes: int           # number of instrument tubes in assembly (-)
    ass_pitch: float       # ass_pitch pitch (cm)
    pin_pitch: float       # pin pitch (cm)
    pin_radius: float      # pin radius (used for average power density in cycle)
    active_height: float   # fuel active height (cm)
    coremap: list          # list of assemblies in the core [i,j] (indexes counting reflector rows)
    source: list           # list of assemblies to be modeled for source definition [i,j]
    ngtubesqtr: int = None # number of guide tubes in quarter assembly (only needed if latsym is 'se')
    nitubesqtr: int = None # number of instrument tubes in quarter assembly (only needed if latsym is 'se')

    # --- computed / parsed outputs ---
    coordinates: List[Tuple[int, int, int, int, float, float]] = field(default_factory=list)
    coordinatesVERA_with_index: List[Tuple[int, int, int, int, float, float, int]] = field(default_factory=list)
    z_core: List[float] = field(default_factory=list)          
    meshheight: np.ndarray = field(default_factory=lambda: np.array([]))  
    nodevolume: List[float] = field(default_factory=list)
    nfuelpins: int = field(init=False)

    def __post_init__(self):
        # compute the number of fuel pins per assembly
        self.nfuelpins = self.npin * self.npin - (self.ngtubes + self.nitubes)

    def compute_radial_coordinates_PARCS(self) -> None:
        """
        Compute (i,j,m,n,x_pin,y_pin) for each pin inside each ass_pitch.
        Indices are 1-based to match typical PARCS-style indexing.
        """
        print("Computing core PARCS coordinates ...")

        self.coordinates.clear()

        a_mid = (self.nass + 1) // 2
        p_mid = (self.npin + 1) // 2

        for i in range(1, self.nass + 1):
            for j in range(1, self.nass + 1):
                x_core = (j - a_mid) * self.ass_pitch
                y_core = (a_mid - i) * self.ass_pitch

                for m in range(1, self.npin + 1):
                    for n in range(1, self.npin + 1):
                        x_pin = x_core + (n - p_mid) * self.pin_pitch
                        y_pin = y_core + (p_mid - m) * self.pin_pitch
                        self.coordinates.append((i, j, m, n, x_pin, y_pin))

    def read_axial_mesh_from_parcs(
        self,
        filepath: Union[str, Path], # users can pass str or Path
        header_token: str = " Number      (cm)      (cm)  Integ.   Assm."
    ) -> None:
        """
        Parse axial mesh information from a PARCS text file section that begins with `Number (cm) (cm) Integ. Assm.`.

        - Reads exactly `naxial_nodes` rows after the header.
        - Flips heights to bottom-to-top order (assuming the code provides this info as top-to-bottom).
        - Computes node volumes: ass_pitch^2 * height.
        """
        filepath = Path(filepath)

        self.z_core.clear()
        self.nodevolume.clear()
        mesh_h: List[float] = []

        lines = filepath.read_text().splitlines()

        found_axial = False
        count = 0

        for line in lines:
            if header_token in line:
                found_axial = True
                count = 0
                continue

            if not found_axial:
                continue

            # Stop when the count exceeds the number of axial nodes
            if count >= self.naxial:
                found_axial = False
                break

            parts = line.split()
            # Parse only if there are three columns
            if len(parts) < 3:
                # If PARCS has blank lines or separators inside the table, we skip
                continue

            try:
                z_val = float(parts[1])
                h_val = float(parts[2])
            except ValueError:
                # Non-numeric line inside the table: skip
                continue

            self.z_core.append(z_val)
            mesh_h.append(h_val)
            count += 1

        if count != self.naxial:
            raise ValueError(
                f"Axial mesh parsing found {count} nodes, expected {self.naxial}. "
                f"Check header_token and file format: {filepath}"
            )

        # Flip heights to bottom-to-top order (your original behavior)
        self.meshheight = np.flip(np.array(mesh_h, dtype=float))

        # Node volumes (real node volume)
        a2 = (self.ass_pitch ** 2)
        self.nodevolume = [a2 * h for h in self.meshheight]

    def compute_2D_coordinates_VERA_qtr(self) -> None:

        print("Computing core VERA coordinates ...")

        self.coordinates.clear() 

        a_mid = (self.nass + 1) // 2
        p_mid = (self.npin + 1) // 2
        
        for i in range(1, self.nass + 1):
            for j in range(1, self.nass + 1):
                x_core = (j - a_mid) * self.ass_pitch
                y_core = (a_mid - i) * self.ass_pitch
                for m in range(1, self.npin + 1):
                    for n in range(1, self.npin + 1):
                        x_pin = x_core + (n - p_mid) * self.pin_pitch
                        y_pin = y_core + (p_mid - m) * self.pin_pitch
                        self.coordinates.append((i, j, m, n, x_pin, y_pin))

        # This section maps classic assembly indices with VERA southeast quarter core indices
        coremap= []
        coremap_quarter= []

        for i in range(2,self.nass):
            for j in range(2,self.nass): 
                coremap.append([i,j])
                
        # Select only the south east corner of this
        for assembly in coremap:
            if assembly[0] > self.nass // 2 and assembly[1] > self.nass //2:
                coremap_quarter.append(assembly)

        # numerate south east corner from 1 to len(coremap_quarter)
        connect = []
        i= 0
        for assembly in coremap_quarter:
            if assembly in self.coremap:  # only consider assemblies that are actually in the core
                connect.append([int(assembly[0]), int(assembly[1]), i])
                i+=1

        # convert to pandas for easier handling
        connect= pd.DataFrame(connect, columns=['x_index','y_index','vera_index'])

        # Now map the pin coordinates to VERA indices
        for coord in self.coordinates:
            # This is valid only for the quarter bottom right of the core
            if ([coord[0],coord[1]] in self.source):  
                for i in range(len(connect)):
                    if (connect.iloc[i, 0] == coord[0]) and (connect.iloc[i, 1] == coord[1]):
                        vera_index = connect.iloc[i, 2]
                        self.coordinatesVERA_with_index.append((coord[0], coord[1], coord[2], coord[3], coord[4], coord[5], vera_index))
                        print(f'Assembly {coord[0]},{coord[1]} has VERA index {vera_index}')
                        break

    def compute_3D_coordinates_VERA_qtr(self,
        filepath: Union[str, Path], # users can pass str or Path
        axial_header_token: str = "axial_edit_bounds"
        ) -> None:

        print("Computing core VERA coordinates ...")

        self.coordinates.clear() 

        a_mid = (self.nass + 1) // 2
        p_mid = (self.npin + 1) // 2
        
        for i in range(1, self.nass + 1):
            for j in range(1, self.nass + 1):
                x_core = (j - a_mid) * self.ass_pitch
                y_core = (a_mid - i) * self.ass_pitch
                for m in range(1, self.npin + 1):
                    for n in range(1, self.npin + 1):
                        x_pin = x_core + (n - p_mid) * self.pin_pitch
                        y_pin = y_core + (p_mid - m) * self.pin_pitch
                        self.coordinates.append((i, j, m, n, x_pin, y_pin))

        # This section maps classic assembly indices with VERA southeast quarter core indices
        coremap= []
        coremap_quarter= []

        for i in range(2,self.nass):
            for j in range(2,self.nass): 
                coremap.append([i,j])
                
        # Select only the south east corner of this
        for assembly in coremap:
            if assembly[0] > self.nass // 2 and assembly[1] > self.nass //2:
                coremap_quarter.append(assembly)

        # numerate south east corner from 1 to len(coremap_quarter)
        connect = []
        i= 0
        for assembly in coremap_quarter:
            if assembly in self.coremap:  # only consider assemblies that are actually in the core
                connect.append([int(assembly[0]), int(assembly[1]), i])
                i+=1

        # convert to pandas for easier handling
        connect= pd.DataFrame(connect, columns=['x_index','y_index','vera_index'])

        # Extract axial mesh from the specfied file
        self.z_core.clear()

        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            lines= f.readlines()

        found_axial= False
        print('Axial Z core:')
        for line in lines:
            if axial_header_token in line:
                found_axial = True
                i = -1
                data= line.split()
                self.z_core= [float(x) for x in data[1:]]
                print(self.z_core)
                found_axial= False
                break

        #flip heights to have the bottom to top order
        self.meshheight= np.diff(self.z_core)
        print('Mesh heights:')
        print(self.meshheight)
        
        ### CURRENTLY NODE VOLUME IS NOT USED IN VERA
        for h in self.meshheight:
            self.nodevolume.append((self.ass_pitch**2)*h) # real node volume

        # Generate full 3D coordinates with VERA index    
        self.coordinatesVERA_with_index.clear() # This will store the coordinates with VERA index
        for coord in self.coordinates:
            # This is valid only for the quarter bottom right of the core
            # The coordinates are in the form (i, j, m, n, x_pin, y_pin)
            if ([coord[0],coord[1]] in self.source):  # Check if the assembly is in the source list
                for i in range(47):
                    if (connect.iloc[i, 0] == coord[0]) and (connect.iloc[i, 1] == coord[1]):
                        vera_index = connect.iloc[i, 2]
                        for k in range(self.naxial-2): # HARD CODED: you have to get rid of the reflectors
                            znode= (self.z_core[k] + self.z_core[k+1])/2
                            self.coordinatesVERA_with_index.append((coord[0], coord[1], coord[2], coord[3], coord[4], coord[5], vera_index, znode, k))
                        print(f'Assembly {coord[0]},{coord[1]} has VERA index {vera_index}')
                        break