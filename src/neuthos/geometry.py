# src/neuthos/geometry.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

@dataclass
class Geometry:
    """
    Geometry class for nuclear reactor core calculations.
    
    This class manages geometry data and computations for a nuclear reactor core,
    including radial and axial coordinate generation, mesh handling, and PARCS/VERA
    coordinate system transformations.
    
    Attributes
    ----------
    naxial : int
        Number of axial nodes (-)
    nass : int
        Number of assemblies in core row/column (-)
    npin : int
        Number of pins in assembly row/column (-)
    latsym : str
        Lattice symmetry: 'full' for full core or 'se' for southeast quarter
    ngtubes : int
        Number of guide tubes in assembly (-)
    nitubes : int
        Number of instrument tubes in assembly (-)
    ass_pitch : float
        Assembly pitch (cm)
    pin_pitch : float
        Pin pitch (cm)
    pin_radius : float
        Pin radius used for average power density calculation in cycle (cm)
    active_height : float
        Fuel active height (cm)
    coremap : list
        List of assemblies in the core as [i,j] coordinates (indexes counting reflector rows)
    source : list
        List of assemblies to be modeled for source definition [i,j]
    ngtubesqtr : int, optional
        Number of guide tubes in quarter assembly (only needed if latsym is 'se')
    nitubesqtr : int, optional
        Number of instrument tubes in quarter assembly (only needed if latsym is 'se')
    coordinates : List[Tuple[int, int, int, int, float, float]]
        Radial coordinates in format (i, j, m, n, x_pin, y_pin)
    coordinatesVERA_with_index : List[Tuple[int, int, int, int, float, float, int]]
        VERA coordinates with index in format (i, j, m, n, x_pin, y_pin, vera_index)
    z_core : List[float]
        Axial core heights (cm)
    meshheight : np.ndarray
        Mesh heights for each axial node (cm)
    nodevolume : List[float]
        Volume of each node (cm³)
    nfuelpins : int
        Computed number of fuel pins per assembly
    """
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

    def compute_2D_coordinates_VERA_qtr(
        self,
        filepath: Union[str, Path] = None,
        axial_header_token: str = "axial_edit_bounds",
        ) -> None:
        """
        Compute 2D radial coordinates for VERA southeast quarter core with VERA indexing.

        This method generates pin coordinates (i, j, m, n, x_pin, y_pin) for all pins in the core,
        then maps assemblies in the southeast quarter to VERA indices. Only pins belonging to
        assemblies in the source list are included in the output with their VERA indices.

        The method:
        - Computes full 2D pin coordinates in PARCS format
        - Identifies southeast quarter assemblies from the full core map
        - Creates a mapping between assembly positions and VERA indices
        - Stores coordinates with VERA indices in coordinatesVERA_with_index
        - Populates the single-node axial mesh (z_core, meshheight, nodevolume) so that the
          2D source writer can place the source slab consistently with the 3D case

        Parameters
        ----------
        filepath : Union[str, Path], optional
            Path to the VERA input file (.inp). When provided, the axial extent of the single
            2D node is read from the `axial_header_token` card (e.g. `axial_edit_bounds 0.0 1.0`
            for a 2D VERA case) and collapsed to a single node spanning [first, last] bound, so
            that the source slab matches the axial extent of the VERA/Serpent 2D geometry.
            When omitted, the node spans [0, active_height] using the Geometry active_height.
        axial_header_token : str, optional
            Header token identifying the axial mesh line in the input file. Default is
            "axial_edit_bounds" (same token as compute_3D_coordinates_VERA_qtr).

        Raises
        ------
        None

        Notes
        -----
        Only assemblies listed in self.source are processed for VERA indexing.
        The VERA indexing starts from 0 for the first southeast quarter assembly.
        The 2D case has a single axial node, so z_core holds two edges [z_bot, z_top] and
        meshheight holds a single value (z_top - z_bot).
        """

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

        # Populate the single-node axial mesh (core height) so the 2D source writer has z_core /
        # meshheight, consistently with compute_3D_coordinates_VERA_qtr.
        self.z_core.clear()
        self.nodevolume.clear()

        z_bot, z_top = None, None
        if filepath is not None:
            filepath = Path(filepath)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if axial_header_token in line:
                    bounds = [float(x) for x in line.split()[1:]]
                    if len(bounds) >= 2:
                        # collapse to a single 2D node spanning the full axial extent
                        z_bot, z_top = bounds[0], bounds[-1]
                    break

        if z_bot is None:
            # no file (or malformed card): fall back to [0, active_height]
            z_bot, z_top = 0.0, self.active_height
            print(f'Axial mesh not read from file: using single node [0, active_height] = [0, {self.active_height}]')
        else:
            print(f'Axial mesh (2D single node) read from {axial_header_token}: [{z_bot}, {z_top}]')

        # z_core holds the two node edges; meshheight the single node height
        self.z_core = [z_bot, z_top]
        self.meshheight = np.diff(self.z_core)
        self.nodevolume = [(self.ass_pitch ** 2) * h for h in self.meshheight]
        print('Mesh heights:')
        print(self.meshheight)

    def compute_3D_coordinates_VERA_qtr(self,
        filepath: Union[str, Path], # users can pass str or Path
        axial_header_token: str = "axial_edit_bounds"
        ) -> None:
        """
        Compute 3D radial and axial coordinates for VERA southeast quarter core with VERA indexing.
        
        This method generates pin coordinates (i, j, m, n, x_pin, y_pin, vera_index, z_node, k) 
        for all pins in the core, then maps assemblies in the southeast quarter to VERA indices 
        and axial node information. Only pins belonging to assemblies in the source list are 
        included in the output with their VERA indices and axial coordinates.
        
        The method:
        - Computes full 3D pin coordinates in PARCS format
        - Extracts axial mesh information from a specified file
        - Identifies southeast quarter assemblies from the full core map
        - Creates a mapping between assembly positions and VERA indices
        - Stores coordinates with VERA indices and axial information in coordinatesVERA_with_index
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the input file containing axial mesh information (typically a VERA input file).
            Users can pass either a string or pathlib.Path object.
        axial_header_token : str, optional
            Header token to identify the line containing axial mesh data in the input file.
            Default is "axial_edit_bounds".
        
        Returns
        -------
        None
            Populates the internal attributes:
            - coordinates: List of 2D pin coordinates (i, j, m, n, x_pin, y_pin)
            - z_core: List of axial heights (cm)
            - meshheight: numpy array of mesh heights for each axial node (cm)
            - nodevolume: List of node volumes (cm³)
            - coordinatesVERA_with_index: List of 3D pin coordinates with VERA index 
              and axial information (i, j, m, n, x_pin, y_pin, vera_index, znode, k)
        
        Raises
        ------
        FileNotFoundError
            If the specified filepath does not exist.
        ValueError
            If the axial_header_token is not found in the file or if axial mesh data is malformed.
        
        Notes
        -----
        - Only assemblies listed in self.source are processed for VERA indexing.
        - The VERA indexing starts from 0 for the first southeast quarter assembly.
        - Axial node volumes are computed as ass_pitch² × mesh_height but are not currently used in VERA calculations.
        - Hard-coded loop excludes first and last axial nodes (reflectors): range(self.naxial-2)
        - The axial node center (znode) is computed as the midpoint between consecutive z_core values.
        
        Examples
        --------
        >>> geom = Geometry(...)
        >>> geom.compute_3D_coordinates_VERA_qtr('vera_input.txt', axial_header_token='axial_edit_bounds')
        """

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