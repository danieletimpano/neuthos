# src/neuthos/cycle.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import h5py
import re
import os
import math

@dataclass
class Cycle:
    """
    Cycle class for managing nuclear reactor cycle information and depletion data.
    
    This class handles the extraction and storage of cycle-related data from Polaris/PARCS and VERA.
    It manages both user inputs and computed/parsed outputs for cycle analysis.
    
    Attributes
    ----------
    Input Attributes:
        nassembly_with_reflectors : int
            Total number of assemblies including reflectors in the core.
        polarisoption : int
            Polaris data usage option:
            - 0: uses literature assumptions for nubar/sigma_f/Er
            - 1: uses Polaris output for nubar and Er
            - 2: uses Polaris output only for Chi
            - 3: uses Polaris output for nubar, Er and Chi
        nsteps : int
            Number of depletion steps in the cycle.
        asspower : float, optional
            Assembly power in MW. Used only if polarisoption is 0 or 1 (default: 0.0).
        groups : int, optional
            Number of energy groups in Polaris finegroup output (default: 252).
            Input is ignored if polarisoption is 0 or 1.
        interpoption : bool, optional
            Whether to perform power interpolation (default: False).
            Input is ignored if polarisoption is 0 or 1.
        interpolnodes : int, optional
            Number of nodes to use for power interpolation (default: None).
            Option only possible for pin power source.
    
    Output/Computed Attributes:
        cycleinfopow : List[float]
            Power levels [MW] for each step in the cycle.
        cycleinfodays : List[float]
            Number of days for each step in the cycle.
        cycleinfoexp : List[float]
            Exposure [GWD/MTIHM] for each step in the cycle.
        coolant_density : dict
            Coolant density [g/cm³] mapped by step index to arrays of shape (naxial, nassembly_with_reflectors).
        assyaxial : List[Tuple[str, int, List[int]]]
            List of tuples containing (type, assy_type, axial_config) for each assembly type.
            Type is either 'FUEL' or 'REFL'.
        assyradial : List[Tuple[int, int, int]]
            List of tuples containing (x_index, y_index, value) for radial assembly configuration.
        polariswidth : List[float]
            Energy group widths [MeV] for Polaris spectrum.
        lattype : List[Tuple[int, str, str, float, str, str, str]]
            List of tuples containing lattice feature data (latfeat).
        refllat : List[int]
            List of reflector lattice type indices.
        fuellat : List[int]
            List of fuel lattice type indices.
        latburn : List[float]
            Burnup levels [GWD/MTIHM] for lattice composition.
        latcomp : List[Tuple]
            List of tuples containing lattice composition data.
        assyexp : List[Tuple[int, int, int]]
            List of tuples containing (x_index, y_index, assigned_depletion_index) for assemblies.
        corevol : float
            Core volume [cm³].
        avgpowdens : float
            Average power density [MeV/cm³].
        exposure : np.ndarray
            3D array of shape (naxial, nassembly_with_reflectors, nsteps) containing assembly exposure values [GWD/MTIHM].
        asspowerdata : List[Tuple[int, int, int, int, float]]
            List of assembly power tuples (step, x_index, y_index, axial_plane, power_value).
        pinpowerdata : List[Tuple[int, int, int, int, int, int, float]]
            List of pin power tuples (step, i_index, j_index, k_index, x_index, y_index, power_value).
        pinpowerdatainterp : List[Tuple[int, int, int, int, int, int, float]]
            List of interpolated pin power tuples (step, i_index, j_index, k_index, x_index, y_index, power_value).
        pinpowerdata2D : List[Tuple[int, int, int, int, int, float]]
            List of 2D pin power tuples (step, i_index, j_index, x_index, y_index, power_value).
        pinpowerdataVERA : List[Tuple]
            List of 3D VERA pin power tuples (step, vera_index, i_index, j_index, k_index,
            m_index, n_index, x_pin, y_pin, power_value, normalized_power) extended with
            (U235, U238, Pu239, Pu241) number densities when explicitdeploption is True.
        pinpowerdataVERA2D : List[Tuple]
            List of 2D VERA pin power tuples (step, vera_index, i_index, j_index, k_index,
            m_index, n_index, x_pin, y_pin, power_value, normalized_power).
    """
    # --- cycle inputs ---
    nassembly_with_reflectors: int                  # total number of assemblies including reflectors
    polarisoption: int                              # 0 uses literature assumptions for nubar/sigma_f/Er, 1 uses Polaris output for nubar and Er, 2 uses Polaris ouput only for Chi, 3 uses Polaris output for nubar, Er and Chi
    nsteps: int                                     # number of depletion steps in the cycle
    asspower : float = 0.0                          # assembly power in MW (used only if polarisoption is 0 or 1)
    groups: int = None                              # number of energy groups in Polaris finegroup output (default: 252) - input is ignored if polarisoption is 0 or 1
    interpoption: bool = False                      # whether to perform power interpolation (default: False) - input is ignored if polarisoption is 0 or 1 
    interpolnodes: int = None                       # number of nodes to use for power interpolation (default: None) -- option only possible for pin power source
    explicitdeploption: bool = True                # whether the MPACT explicit depletion module was used (VERA only): if True, pin-wise U-235/U-238/Pu-239/Pu-241 densities are read alongside the pin powers

    # --- computed / parsed outputs ---
    cycleinfopow: List[float] = field(default_factory=list)
    cycleinfodays: List[float] = field(default_factory=list)
    cycleinfoexp: List[float] = field(default_factory=list)
    coolant_density: dict = field(default_factory=dict)
    assyaxial: List[Tuple[str, int, List[int]]] = field(default_factory=list)                # List of tuples (type, assy_type, axial_config)
    assyradial: List[Tuple[int, int, int]] = field(default_factory=list)                     # List of tuples (x_index, y_index, value)
    polariswidth: List[float] = field(default_factory=list)                                  # Energy group widths for Polaris
    lattype: List[Tuple[int, str, str, float, str, str, str]] = field(default_factory=list)  # List of tuples (latfeat)
    refllat: List[int] = field(default_factory=list)                                         # List of reflector lattice types
    fuellat: List[int] = field(default_factory=list)                                         # List of fuel lattice types
    latburn: List[float] = field(default_factory=list)                                       # Burnup levels for lattice composition
    latcomp: List[Tuple] = field(default_factory=list)                                       # List of tuples (latcomp)
    assyexp: List[Tuple[int, int, int]] = field(default_factory=list)                        # List of tuples (x_index, y_index, assigned_depletion_index)
    corevol: float = field(init=0)                                                           # Core volume in cm3
    avgpowdens: float  = field(init=0)                                                       # Average power density in MW/cm3
    exposure: np.ndarray = field(init=False)                                                 # 3D array (naxial, nassembly_with_reflectors, nsteps) of assembly exposure values
    asspowerdata: List[Tuple[int, int, int, int, float]] = field(default_factory=list)       # List of tuples (step, x_index, y_index, axial_plane, power_value)
    pinpowerdata: List[Tuple[int, int, int, int, int, int, float]] = field(default_factory=list) # List of tuples (step, i_index, j_index, k_index, x_index, y_index, power_value)
    pinpowerdatainterp: List[Tuple[int, int, int, int, int, int, float]] = field(default_factory=list) # List of tuples (step, i_index, j_index, k_index, x_index, y_index, power_value) for interpolated power values
    pinpowerdata2D: List[Tuple[int, int, int, int, int, float]] = field(default_factory=list) # List of tuples (step, i_index, j_index, x_index, y_index, power_value) for 2D pin power data
    pinpowerdataVERA: List[Tuple] = field(default_factory=list)                              # List of tuples (step, vera_index, i, j, k, m, n, x_pin, y_pin, power_value, normalized_power[, U235, U238, Pu239, Pu241]) for 3D VERA pin power data
    pinpowerdataVERA2D: List[Tuple] = field(default_factory=list)                            # List of tuples (step, vera_index, i, j, k, m, n, x_pin, y_pin, power_value, normalized_power) for 2D VERA pin power data

    # PARCS related methods

    def extract_cycle_info(
            self, filepath: Union[str, Path]) -> None: 
        """
        Extract general cycle information from a PARCS depletion summary file.
        The information extracted includes power levels, days, and exposure for each step in the cycle.
        """
        
        if filepath.endswith('.parcs_dpl'): 
            print(f'Extracting cycle information from file: {filepath}')
            
            with open(filepath, 'r') as f:
                file_content = f.read()

            file_content_statepoint = file_content.find('   PT   RE     Keff  Power  AxOff     Pz    Pxy   Pxyz   PPin    Days B(GW/T)    Bmax   Beta  notch    ppm  Tf(K)  Tm(K)  d(g/cc) void(%) Xe(1/cc)  Sm(1/cc)     Fdh      Fq    CHFR', 0)  
            file_content_end = file_content.find('_______________________________________________________________________________', 0)
            summary_content = file_content[file_content_statepoint:file_content_end].split('\n')[:]
            #print(summary_content)

            # Loop through the lines to extract the numerical data
            for line in summary_content:
                # Split the line into columns
                cols = line.split()
                # Ignore header lines and process only rows with numerical data
                if len(cols) > 0: # avoid empty lines
                    if cols[0].isdigit():  # The first value in a valid row should be a number
                        self.cycleinfopow.append(float(cols[3]))
                        self.cycleinfodays.append(float(cols[9]))
                        self.cycleinfoexp.append(float(cols[10]))

            print('The cycle Power Levels are:')
            print(self.cycleinfopow)
            print('The cycle corresponding days are:')
            print(self.cycleinfodays)
            print('The cycle corresponding exposure are:')
            print(self.cycleinfoexp)

        elif filepath.endswith(".h5"):
            print(f'Extracting cycle information from HDF5 file: {filepath}')

            self.cycleinfopow = []
            self.cycleinfodays = []
            self.cycleinfoexp = []

            with h5py.File(filepath, 'r') as h5file:
                for step in range(self.nsteps):
                    print(f'Extracting data for step {step+1} from HDF5 file...')
                    self.cycleinfopow.append(float(h5file[f'STATE_{step+1:04d}/power'][()]))
                    self.cycleinfodays.append(float(h5file[f'STATE_{step+1:04d}/exposure_efpd'][()]))
                    self.cycleinfoexp.append(float(h5file[f'STATE_{step+1:04d}/exposure'][()]))

            print('The cycle Power Levels are:')
            print(self.cycleinfopow)
            print('The cycle corresponding days are:')
            print(self.cycleinfodays)
            print('The cycle corresponding exposure are:')
            print(self.cycleinfoexp)

    def extract_coolant_info(
            self,  geom: "Geometry", filepath: Union[str, Path]) -> None:
        """
        Extracts coolant information that varies during the cycle from a PARCS cycle summary file.
        The information extracted includes coolant density for each assembly in the core, including reflector assemblies.
        """

        print('Extra: extracting coolant information')
        self.coolant_density = {}

        with open(filepath, 'r') as f:
            file_content = f.read()

        file_content_end = 0
        for i in range(len(self.cycleinfopow)):
            file_content_statepoint = file_content.find('   PT   RE     Keff  Power  AxOff     Pz    Pxy   Pxyz   PPin    Days B(GW/T)    Bmax   Beta  notch    ppm  Tf(K)  Tm(K)  d(g/cc) void(%) Xe(1/cc)  Sm(1/cc)     Fdh      Fq    CHFR', file_content_end if i > 0 else 0)
            file_content_end = file_content.find('   PT   RE     Keff  Power  AxOff     Pz    Pxy   Pxyz   PPin    Days B(GW/T)    Bmax   Beta  notch    ppm  Tf(K)  Tm(K)  d(g/cc) void(%) Xe(1/cc)  Sm(1/cc)     Fdh      Fq    CHFR', file_content_statepoint + 1)
            statepoint_content = file_content[file_content_statepoint:file_content_end]
            
            # Process the statepoint_content 
            # Find the beginning of the section "DCO 3D MAP"
            density_section_start = statepoint_content.find("DCO 3D MAP")
            if density_section_start == -1:
                raise ValueError("Sezione 'DCO' non trovata")

            # Each subsection features 10 assemblies, there is one header and one empty line between subsections
            complete_boxes = self.nassembly_with_reflectors // 10
            partial_columns = self.nassembly_with_reflectors % 10
            section_boundary = (complete_boxes)*(geom.naxial+2) + (partial_columns)*(geom.naxial + 2)

            density_section = statepoint_content[density_section_start:].split('\n')[:(section_boundary)]
            
            # Initialize an empty list to hold the matrix rows
            matrix = []
            final = np.zeros((geom.naxial, self.nassembly_with_reflectors))
            
            # Loop through the lines to extract the numerical data
            for line in density_section:
                # Split the line into columns
                cols = line.split()
                # Ignore header lines and process only rows with numerical data
                if len(cols) > 0: # avoid empty lines
                    if cols[0].isdigit():  # The first value in a valid row should be a number
                        # Extract the numerical values starting from the second column
                        numbers = [float(val) for val in cols[1:]]
                        for number in numbers:
                            matrix.append(number)
            # Reshape the flattened data into a naxial x nassembly_with_reflectors matrix
            for j in range(complete_boxes):
                final[:,(j*10):((j+1)*10)]= np.array(matrix[j*geom.naxial*10:(j+1)*geom.naxial*10]).reshape((geom.naxial, 10))
            if partial_columns != 0: # CHECK WHY THIS IS NOT WORKING
                final[:,complete_boxes:complete_boxes+partial_columns+1] = np.array(matrix[complete_boxes*geom.naxial*10-1:complete_boxes*geom.naxial*10+partial_columns*geom.naxial]).reshape((geom.naxial,partial_columns))
            # save the matrix in a dictionary with the corresponding i 
            self.coolant_density[i] = final

    def extract_assyaxial(
            self, filepath: Union[str, Path]) -> None:
        """
        Extracts assembly axial configuration from a PARCS assembly geometry file.
        """

        print('Extracting assembly axial configuration ...')

        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if "assy_type " in line:
                if 'FUEL' in line:
                    parts = line.split()
                    assy_type = int(parts[1])
                    assy_geom= []
                    for part in parts[2:]:
                        if '*' in part:
                            count, value = part.split('*')
                            assy_geom.extend([int(value)] * int(count))
                        else:
                            continue
                    self.assyaxial.append(['FUEL',assy_type, assy_geom[::-1]])
                elif 'REFL' in line: 
                    parts = line.split()
                    assy_type = int(parts[1])
                    assy_geom= []
                    for part in parts[2:]:
                        if '*' in part:
                            count, value = part.split('*')
                            assy_geom.extend([int(value)] * int(count))
                        else:
                            continue
                    self.assyaxial.append(['REFL', assy_type, assy_geom[::-1]]) # NOTE: reverse the list to have the latttice from top to bottom, REASON: burnup profiles are provided from top to bottom

    def extract_assyradial(
            self, geom: "Geometry", filepath: Union[str, Path]) -> None:
        """
        Extracts assembly radial configuration from a PARCS regular coremap input file.
        """

        print('Extracting assembly radial configuration ...')

        with open(filepath, 'r') as f:
            data = f.readlines()
        
        data_str = ''.join(data)
        cases = re.findall(r'rad_conf.*?file', data_str, re.DOTALL)
        cases = [case for case in cases if "rad_conf" in case]

        if not cases:
            print("No radial configuration found.")
        else:
            print(f'Number of radial configurations found: {len(cases)}')

        for case in cases:
            lines = case.strip().split('\n')
            x_index= 0
            y_indices = list(range(1, geom.nass + 1))
            for line in lines[1:geom.nass+1]:
                values = list(map(int, line.split()))
                x_index += 1
                for y_index, value in zip(y_indices, values):
                        self.assyradial.append((x_index, y_index, value))

    def extract_exposure(
            self, geom: "Geometry", filepath: Union[str, Path]) -> None:
        """

        Extracts assembly exposure data from a PARCS depletion summary file.
        """

        # initialize exposure array
        self.exposure = np.zeros(
            (
                geom.naxial,
                self.nassembly_with_reflectors,
                self.nsteps,
            ),
            dtype=float
        )

        print('Extracting assembly exposure data ...')

        # take all the assemblies
        assy= self.assyradial
        assy_sel= []
        j= 1
        # remove from this list the assemblies that have value = 0
        for i in range(len(assy)):
            if assy[i][2] != 0:
                assy_sel.append((assy[i][0], assy[i][1], j))  # assign to each assembly a number going from 1 to the number of assemblies remaining in the list
                j += 1
        # save the list class property
        self.assyexp= assy_sel

        # open file for assembly exposure
        with open(filepath, 'r') as f:
            lines = f.readlines()
            found_exposure= False
            found_matrix= False
            i= 0
            k= 0
            n= 0
            m= -1
        for line in lines:
            if (' EXP 3D MAP 1.0E+00' in line) and (found_exposure== False) and (found_matrix== False):
                found_exposure= True
                i= 0 # line index 
                k= 0 # column shifter
                n= 0 # block index
                m+=1 # burnup step index
                print('found exposure!')
            
            elif (found_exposure == True) and (found_matrix== False) and ('k lb' in line):
                found_matrix= True
                i= 0

            elif (found_exposure== True) and (found_matrix== True) and (i<geom.naxial) and (n+k<(self.nassembly_with_reflectors-1)):
                parts= line.split()
                for k in range(10):
                    self.exposure[i,n + k,m] = float(parts[1 + k])
                i += 1

            elif (found_exposure== True) and (found_matrix== True) and (i==geom.naxial) and (n+k<(self.nassembly_with_reflectors-1)):
                found_matrix= False
                i= 0
                n += 10
                k= 9

            elif (found_exposure== True) and (found_matrix== True) and (i==0) and (n+k==(self.nassembly_with_reflectors+8)): # ERROR: it should be +8 not + 9
                found_exposure= False
                found_matrix= False

    def extract_asspower(self, geom: "Geometry", out: "Outputs", folderpath: Union[str, Path]) -> None:
        """

        Extracts assembly power data from a PARCS depletion summary file.
        """

        print('The average power density is:')
        self.corevol = geom.nfuelpins * math.pi * (geom.pin_radius)**2 * (geom.active_height)        # cm3
        self.avgpowdens= self.asspower / (self.corevol)                                              # W/cm3
        print(self.avgpowdens)                                                                       # W/cm3

        print('The axial mesh heights are:')
        print(geom.meshheight)

        print('The power level vector is:')
        print(self.cycleinfopow)

        print('Extracting assembly power data ...')
        c = 0 # timestep counter for printing 
        d = 0 # assembly counter for printing

        for file in os.listdir(folderpath):
            # check files which contain parcs_pin in the name
            if ".parcs_3dpw" in file:

                # open the file and save all the lines
                with open(os.path.join(folderpath, file), 'r') as f:
                    data = f.readlines()

                # find all the lines in the file between two time cases
                data_str = ''.join(data)
                cases = re.split(r'(?=Time Step:)', data_str)
                cases = [case for case in cases if "Time Step:" in case]
                if not cases:
                    print("No power found.")
                else:
                    c += 1
                    print(f'... Extracting Timesteps ... ')

                for case in cases:

                    lines = case.strip().split('\n')
                    time_info_vec = lines[0].split()
                    case_number = int(time_info_vec[2])  # save timestep point (burnup point)

                    # find all the lines in the file between two time cases
                    subcases = re.split(r'(?=Distribution at Plane)', case)
                    subcases = [subcase for subcase in subcases if "Distribution at Plane" in subcase]
                    if not cases:
                        print("No power found.")
                    else:
                        d += 1
                        print(f'... Extracting Timestep ...  ' + str(d))

                    for subcase in subcases:

                        lines = subcase.strip().split('\n')
                        axial_plane_vec = lines[0].split()
                        axial_plane = int(axial_plane_vec[3])  # axial plane power distribution
                        print(f'... Extracting Axial Plane ...  ' + str(axial_plane))
                        # Skip cases where k = 0 as they do not correspond to an axial layer
                        y_indices = list(map(int, lines[2].split()[:]))
                        y_indices.append(y_indices[-1] + 1)
                        # print(cls.mesheight[axial_plane-1])
                        # print('daje')
                        for line in lines[3:geom.nass+1]:
                            values = list(map(float, line.split()))
                            #center the array values in a bigger array of size y_indices to get the correct position in the array
                            #note: this is necessary because the output file does not contain the zero values
                            x_index = int(values.pop(0))
                            centered_values = [0] * (len(y_indices))
                            start_index = ((len(y_indices)) - len(values)) // 2
                            centered_values[start_index:start_index + len(values)] = values
                            y_indices_new = [y for y, val in zip(y_indices, centered_values) if val != 0]
                            for y_index, value in zip(y_indices_new, values):
                                self.asspowerdata.append((case_number, x_index, y_index, axial_plane, value*self.avgpowdens*geom.nfuelpins*math.pi*(geom.pin_radius)**2*geom.meshheight[axial_plane-1]*(self.cycleinfopow[case_number-1]/100))) 
                                                                                                                                                                                                       
        # SANITY CHECK: total power (check for one burnup step)
        for step in range(self.nsteps):
            total_power= 0
            count = 0
            for power in self.asspowerdata:
                if (power[0] == (step + 1)):
                    count += 1
                    total_power += power[4]
            
            outpath = Path(os.path.join(out.base_dir, 'sanity_checks/ass_source'))
            outpath.mkdir(parents=True, exist_ok=True)

            with open(os.path.join(outpath, f'sanity_check_ass_'+ str(step) + '.txt'), 'w') as f:
                f.write(f'Total power: {total_power}\n')
                f.write(f'Count: {count}')

    def extract_pinpower(self, geom: "Geometry", out: "Outputs", folderpath: Union[str, Path]) -> None:
        """
            Extracts 3D pinpower from PARCS .parcs_pin files.
        """   

        print('The average power density is:')
        self.corevol = geom.nfuelpins * math.pi * (geom.pin_radius)**2 * (geom.active_height)        # cm3
        self.avgpowdens= self.asspower / (self.corevol)                                              # W/cm3
        print(self.avgpowdens)                                                                       # W/cm3

        print('The power level vector is:')
        print(self.cycleinfopow)

        print('Extracting pin power data ...')
        c = 0 # assembly counter for printing 

        for file in os.listdir(folderpath):
            # check files which contain parcs_pin in the name
            if ".parcs_pin" in file:

            # open the file and save all the lines
                with open(os.path.join(folderpath, file), 'r') as f:
                    data = f.readlines()

            # find all the lines in the file between two cases
                data_str = ''.join(data)
                cases = re.split(r'(?=Case:)', data_str)
                cases = [case for case in cases if "Case:" in case]
                if not cases:
                    print("No power found.")
                else:
                    c += 1
                    print(f'... Extracting non-dummy assembly n. ' + str(c))

                for case in cases:
                    lines = case.strip().split('\n')
                    case_info = lines[0].split()
                    case_number = int(case_info[1])  # save burnup point
                    i = int(case_info[9])            # save x core index - check this (there is a different naming between Parcs and our coordinates system)
                    j = int(case_info[8])            # save y core index - check this (there is a different naming between Parcs and our coordinates system)
                    k = int(case_info[10])           # save z core index
                    
                    # Skip cases where k = 0 as they do not correspond to an axial layer but to average, and k = 1 and k = 41 as they correspond to reflectors
                    if (k == 0) or (k == 1) or (k == geom.naxial):
                        continue
                    y_indices = list(map(int, lines[1].split()[:]))
                    for line in lines[2:]:
                        values = list(map(float, line.split()))
                        x_index = int(values.pop(0))
                        for y_index, value in zip(y_indices, values):
                            self.pinpowerdata.append((case_number, i, j, k, x_index, y_index, value*self.avgpowdens*math.pi*(geom.pin_radius)**2*geom.meshheight[k-1]*(self.cycleinfopow[case_number-1]/100),value)) # ALTERNATIVE using node volume considering the whole assembly cls.nodevolume[k-1]/225

        # SANITY CHECK 
        outpath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        for step in range(self.nsteps):
            total_power= 0
            count = 0
            for power in self.pinpowerdata:
                if (power[0] == (step + 1)):
                    count += 1
                    total_power += power[6]
            
            with open(outpath / f'sanity_check_pin_{step}.txt', 'w') as f:
                f.write(f'Total power: {total_power}\n')
                f.write(f'Count: {count}')

        # power interpolation --- NOTE: the interpolation option considers powers as point values, but they are rather node averaged values. This approximation will be solved in future releases, the interpolation option is also not often used.
        if self.interpoption:
            
            print('Interpolating power profiles...')

            for step in range(self.nsteps):

                pin_power_vec = {}
                for pin in self.pinpowerdata:
                    if pin[0] == step:  # take only selected step
                        ijxy_index = (pin[1], pin[2], pin[4], pin[5]) # select the pin
                        if ijxy_index not in pin_power_vec:
                            pin_power_vec[ijxy_index] = []
                        pin_power_vec[ijxy_index].append(pin[7]) # save in relative power vector for that pin, appending each axial layer
                
                # given the interp number you need to find equally spaced nodes between z_core max and z_core min
                # flip z_core data to have planes from bottom to top
                z_core= np.flip(geom.z_core)
                print('Z core coordinates from bottom to top:', z_core)
                z_core= z_core[1:len(geom.z_core)-1] # get rid of bottom and top reflectors
                z_core_interp= np.linspace(np.min(z_core),np.max(z_core),self.interpol) # z coordinates must be monotonically increasing
                #z_core_interp= [17.55, 50, 120, 175, 250, 325, 365]
                meshboundaries= np.zeros(len(z_core_interp)-1)

                for i in range(len(z_core_interp)-1):
                    meshboundaries[i]= (z_core_interp[i+1] + z_core_interp[i])/2
                meshboundaries_all= np.concatenate(([15.24], meshboundaries, [383.4]))
                mesheight_interp= np.zeros(len(meshboundaries_all)-1)
                
                for i in range(len(meshboundaries_all)-1):
                    mesheight_interp[i]= meshboundaries_all[i+1]-meshboundaries_all[i]
                
                mesheight_interp= mesheight_interp #save meshheightinterp
                z_core_interp= z_core_interp #save zcoreinterp

                print('New mesheigths')
                print(mesheight_interp)
                print('Original mesheights')
                print(geom.meshheight)

                # you need to interpolated the previous pin powers
                for ijxy_index, power_values in pin_power_vec.items():
                    pin_power_vec_interp = np.interp(z_core_interp, z_core, np.flip(power_values))
                    for k, z_value in enumerate(z_core_interp):
                        self.pinpowerdatainterp.append((step, ijxy_index[0], ijxy_index[1], k + 1, ijxy_index[2], ijxy_index[3], pin_power_vec_interp[k]*self.avgpowdens*math.pi*(geom.pin_radius)**2*mesheight_interp[k-1]*(self.cycleinfopow[case_number-1]/100), pin_power_vec_interp[k]))

                    # take sample case to plot and check interpolation
                    if ijxy_index == (9,16,9,15): 
                        plt.figure()
                        plt.plot(z_core, np.flip(power_values), '-o', color='black', label='Original')
                        plt.plot(z_core_interp, pin_power_vec_interp, '-x', color='red', label='Interpolated')
                        plt.grid()
                        plt.legend()
                        plt.savefig('interpolation_check.png')

                # SANITY CHECK: total power (check for one burnup step)
                total_power= 0
                count = 0
                for power in self.pinpowerdata:
                    if (power[0] == step):
                        count += 1
                        total_power += power[6]
                
                with open(outpath / f'sanity_check_pin_{step}.txt', 'w') as f:
                    f.write(f'Total power: {total_power}\n')
                    f.write(f'Count: {count}')

                total_power= 0
                count = 0
                for power in self.pinpowerdatainterp:
                    if (power[0] == step):
                        count += 1
                        total_power += power[6]
                
                with open(outpath / f'sanity_check_pin_interp_{step}.txt', 'w') as f:
                    f.write(f'Total power: {total_power}\n')
                    f.write(f'Count: {count}')

    def extract_pinpower2D(self, geom: "Geometry", out: "Outputs", folderpath: Union[str, Path]) -> None:
        """
            Extracts 2D pinpower from PARCS .parcs_pin files.
        """   

        print('The average power density is:')
        self.corevol = geom.nfuelpins * math.pi * (geom.pin_radius)**2 * (geom.active_height)        # cm3
        self.avgpowdens= self.asspower / (self.corevol)                                              # W/cm3
        print(self.avgpowdens)                                                                       # W/cm3

        print('The power level vector is:')
        print(self.cycleinfopow)

        print('Extracting pin power data ...')
        c = 0 # assembly counter for printing 

        for file in os.listdir(folderpath):
            # check files which contain parcs_pin in the name
            if ".parcs_pin" in file:

            # open the file and save all the lines
                with open(os.path.join(folderpath, file), 'r') as f:
                    data = f.readlines()

            # find all the lines in the file between two cases
                data_str = ''.join(data)
                cases = re.split(r'(?=Case:)', data_str)
                cases = [case for case in cases if "Case:" in case]
                if not cases:
                    print("No power found.")
                else:
                    c += 1
                    print(f'... Extracting non-dummy assembly n. ' + str(c))

                for case in cases:
                    lines = case.strip().split('\n')
                    case_info = lines[0].split()
                    case_number = int(case_info[1])  # save burnup point
                    i = int(case_info[9])            # save x core index - check this (there is a different naming between Parcs and our coordinates system)
                    j = int(case_info[8])            # save y core index - check this (there is a different naming between Parcs and our coordinates system)
                    k = int(case_info[10])           # save z core index
                    
                    # Take only the cases where k = 0 as they correspond to the axial average
                    if k == 0:
                        if [i,j] in geom.source: # filter to extract only the sources of interest
                            y_indices = list(map(int, lines[1].split()[:]))
                            for line in lines[2:]:
                                values = list(map(float, line.split()))
                                x_index = int(values.pop(0))
                                for y_index, value in zip(y_indices, values):
                                    if value != 0: # check if the pin power is not zero
                                        self.pinpowerdata2D.append((case_number, i, j, k, x_index, y_index, value*self.avgpowdens*math.pi*(geom.pin_radius)**2*self.cycleinfopow[case_number-1]/100)) 

        # SANITY CHECK 
        outpath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        for step in range(self.nsteps):
            total_power= 0
            count = 0
            for power in self.pinpowerdata2D:
                if (power[0] == (step + 1)):
                    count += 1
                    total_power += power[6]
            
            with open(outpath / f'sanity_check_pin_{step}.txt', 'w') as f:
                f.write(f'Total power: {total_power}\n')
                f.write(f'Count: {count}')

    def extract_lattype(
            self, filepath: Union[str, Path]) -> None:
        """
        Extracts lattice type data from a PARCS pmax.dir file.
        NOTE: this method will be deprecated in the future NEUTHOS release in favour of user specification of reference polaris cases.
        """


        print('Extracting lattice type data from PARCS pmax.dir file')

        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            latfeat= []
            if "PMAXS_F" in line:
                parts = line.split()
                if 'refl' in line:
                    latfeat= [int(parts[1]), parts[5], 'refl', 0.0, 'bpn', 'sgn', '-']
                elif 'fuel' in line:
                    if '00-BP' in line:
                        if 'SG' in line:
                            latfeat= [int(parts[1]), parts[6], 'fuel', float(parts[7]), 'bpn', 'sgy', parts[12]]
                        elif 'SG' not in line:
                            latfeat= [int(parts[1]), parts[6], 'fuel', float(parts[7]), 'bpn', 'sgn', parts[11]]
                    elif '12-BP' in line:
                        if 'SG' in line:
                            latfeat= [int(parts[1]), parts[6], 'fuel', float(parts[7]), 'bpy', 'sgy', parts[12]]
                        elif 'SG' not in line:
                            latfeat= [int(parts[1]), parts[6], 'fuel', float(parts[7]), 'bpy', 'sgn', parts[11]]
                self.lattype.append((latfeat))
        
        for lat in self.lattype: #save refl and fuel assembly types
            if lat[2] == 'refl':
                self.refllat.append(lat[0])
            elif lat[2] == 'fuel':
                self.fuellat.append(lat[0])
        
        print('The fuel lattice types are:')
        print(self.fuellat)
        print('The reflector lattice types are:')
        print(self.refllat)

    # Polaris related methods

    def extract_spectrum(
            self, sp, file_content, ngroups) -> List[Tuple[float, float]]:
        """
        Extracts neutron spectrum from a Polaris (SCALE) spectrum file.
        The spectrum is extracted for a given statepoint (sp) and number of energy groups (ngroups).
        Returns a list of tuples containing (energy_bin_midpoint, chi_value).
        The extracted emission spectrum is used in two of the polarisoption for lattice composition extraction.
        """

        print('Extracting neutron spectrum from Polaris ...')

        # Find the statepoint to extract polaris neutron spectrum
        file_content_statepoint = file_content.find('--------------------------------------------------\n'+ 'Statepoint ' + str(sp)+ '\n' + '--------------------------------------------------\n')
        file_content= file_content[file_content_statepoint:]

        # Extract the energy group structure
        energy_pattern = r'\|\s*(\d+)\s*\|\s*([\d.E+-]+)\s*\|\s*([\d.E+-]+)\s*\|'
        energy_matches = re.findall(energy_pattern, file_content)
        
        group_widths = []
        group_em= []
        middle= []
        for _, upper, lower in energy_matches:
            width = (float(upper) - float(lower)) / 1e6
            em= float(upper) / 1e6 # you need the upper bound of the energy group
            group_widths.append(width)
            group_em.append(em)
        
        #print([float(f"{mid:.5e}") for mid in np.flip(middle)])
        #print('\n')

        self.polariswidth= np.flip(group_widths[:(ngroups-1)])

        # Find the start of the "Fission Macroscopic Cross Sections" section
        fission_section_start = file_content.find("Fission Macroscopic Cross Sections")
        if fission_section_start == -1:
            raise ValueError("Sezione 'Fission Macroscopic Cross Sections' non trovata")

        # Extract Chi values
        chi_pattern = r'\|\s*(\d+)\s*\|.*?\|.*?\|.*?\|\s*([\d.E+-]+)\s*\|'
        # Extract Chi values only for the fission sections
        chi_section = file_content[fission_section_start:].split('\n')[:(ngroups+10)]
        chi_section = '\n'.join(chi_section)
        chi_matches = re.findall(chi_pattern, chi_section)
        
        chi_values = []
        for group, chi in chi_matches:
            group_index = int(group) - 1
            if group_index < len(group_widths):
                chi_per_mev = float(chi) / group_widths[group_index]
                em_bin= group_em[group_index]
                chi_values.append((em_bin, chi_per_mev))

        return chi_values
     
    def extract_latcomp_scale63(
            self, geom: "Geometry", filepath: Union[str, Path], filepath_finegroup: Union[str, Path] = None) -> None:
        """
        Extracts lattice composition from a Polaris (SCALE 6.3) output file, as a function of burnup.
        """

        print('Extracting lattice composition ...')

        if geom.latsym == 'full':
            line_pins_read = geom.npin
            latfuelpins = geom.npin*geom.npin - geom.ngtubes - geom.nitubes
        
        elif geom.latsym == 'se':
            line_pins_read = geom.npin//2
            latfuelpins = (geom.npin//2)**2 - geom.ngtubesqtr - geom.nitubesqtr  

        for lattice in self.lattype:

            if lattice[2] == 'fuel':
                #path to standard polaris benchmark output
                path = os.path.join(filepath, str(lattice[0]) + '_' + lattice[6] + '.out')

                # assign path to fine group polaris output if polarisoption requires it
                if self.polarisoption == 2 or self.polarisoption == 3:
                    filefinegroup= os.path.join(filepath_finegroup, str(lattice[0]) + '_' + lattice[6] + '.out')

                with open(path, 'r') as f:
                    lines = f.readlines()
                    latcomp= [lattice[0]]
                    index= 0
                    nubar= 0
                    found_statepoint= False
                    found_nubar= False
                    found_energy= False
        
                for line in lines:
                    if "bu " in line:
                        parts = line.split()
                        burnup= [float(x) for x in parts[1:]] # burnup is in GWD/MTIHM
                        self.latburn = burnup
                        statepoint = np.linspace(1, len(burnup), len(burnup))
                        found_statepoint = False
                        index= 1
                    if (' Statepoint ' + str(index) + '\n' in line) and (index<=len(burnup)):
                        found_statepoint = True
                        print('State ' + str(index) + ' Burnup ' + str(burnup[index-1]))
                    elif (found_statepoint==True) and '|------|----------|----------|----------|----------|----------|----------|' in line:
                        found_energy= True
                        line_count= 1
                        fluxweightedER= np.zeros((2,2))
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count == 1) or (line_count == 2)):
                        parts= line.split()
                        fluxweightedER[line_count-1,0]= float(parts[3].split('|')[0])
                        line_count += 1
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count == 11) or (line_count == 12)):
                        parts= line.split()
                        fluxweightedER[line_count-11,1]= float(parts[7])
                        line_count += 1
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count == 13)):
                        line_count= 0
                        er= fluxweightedER[0,0]*fluxweightedER[0,1] + fluxweightedER[1,0]*fluxweightedER[1,1]
                        latcomp= np.append(latcomp, [burnup[index-1], er])
                        found_energy= False
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count != 1) or (line_count != 2) or (line_count != 11) or (line_count != 12) or (line_count != 13)):
                        line_count += 1
                    elif (found_statepoint==True) and 'U-235 Concentration' in line:
                        parts= line.split()
                        #print('U5')
                        latcomp= np.append(latcomp,parts[3])
                    elif (found_statepoint==True) and 'U-238 Concentration' in line:
                        parts= line.split()
                        #print('U8')
                        latcomp= np.append(latcomp, parts[3])
                    elif (found_statepoint==True) and 'Pu-239 Concentration' in line:
                        parts= line.split()
                        #print('PU9')
                        latcomp= np.append(latcomp, parts[3])
                    elif (found_statepoint==True) and 'Pu-241 Concentration' in line:
                        parts= line.split()
                        #print('PU41')
                        latcomp= np.append(latcomp, parts[3])
                    elif (found_statepoint==True) and 'Neutrons Per Fission' in line:
                        #print('Nubar')
                        found_nubar= True
                        nubar= 0
                        line_count = 0
                    elif (found_statepoint==True) and (found_nubar==True) and (line_count < 1):
                        line_count += 1
                    elif (found_statepoint==True) and (found_nubar==True) and (line_count >= 1) and (line_count < line_pins_read):
                        parts= line.split()
                        adding= [float(x) for x in parts[:]]
                        nubar= nubar + np.sum(adding)
                        line_count += 1
                    elif (found_statepoint==True) and (found_nubar==True) and (line_count == line_pins_read):
                        parts= line.split()
                        adding= [float(x) for x in parts[:]]
                        nubar= nubar + np.sum(adding)
                        nubar= nubar/latfuelpins
                        latcomp= np.append(latcomp, nubar)
                        line_count= 0
                        found_nubar= False
                        found_statepoint= False
                        print('Lattice nubar is: ' + str(nubar) + ' for statepoint ' + str(index))
                        print('Lattice ER is: ' + str(er)+ ' for statepoint ' + str(index))
                        
                        if (self.polarisoption == 2) or (self.polarisoption == 3): # option 2 and option 3 require spectrum extraction from Polaris output


                            chi_values = self.extract_spectrum(index,filefinegroup,self.groups)
                            ene_vector= []
                            chi_vector= []

                            for ene_group, _ in chi_values:
                                ene_vector.append(ene_group)
                            ene_vector= np.array(ene_vector)

                            for _, chi_group in chi_values:
                                chi_vector.append(chi_group)
                            chi_vector= np.array(chi_vector)

                            latcomp= np.append(latcomp, np.flip(ene_vector))
                            latcomp= np.append(latcomp, np.flip(chi_vector))

                        index += 1

                self.latcomp.append((latcomp))

    def extract_latcomp_scale62(
            self, geom: "Geometry", filepath: Union[str, Path], filepath_finegroup: Union[str, Path] = None) -> None:
        """
        Extracts lattice composition from a Polaris (SCALE 6.2) output file, as a function of burnup.
        """

        print('Extracting lattice composition ...')

        if geom.latsym == 'full':
            line_pins_read = geom.npin
            latfuelpins = geom.npin*geom.npin - geom.ngtubes - geom.nitubes
        
        elif geom.latsym == 'se':
            line_pins_read = geom.npin//2
            latfuelpins = (geom.npin//2)**2 - geom.ngtubesqtr - geom.nitubesqtr  
        for lattice in self.lattype:

            if lattice[2] == 'fuel':
                #path to standard polaris benchmark output
                path = os.path.join(filepath, str(lattice[0]) + '_' + lattice[6] + '.out')
                print('Reading lattice file: ' + path)

                # assign path to fine group polaris output if polarisoption requires it
                if self.polarisoption == 2 or self.polarisoption == 3:
                    filefinegroup= os.path.join(filepath_finegroup, str(lattice[0]) + '_' + lattice[6] + '.out')

                with open(path, 'r') as f:
                    lines = f.readlines()
                    latcomp= [lattice[0]]
                    index= 0
                    nubar= 0
                    found_statepoint= False
                    found_nubar= False
                    found_energy= False
                    found_startresult= False
        
                for line in lines:
                    if "bu " in line:
                        parts = line.split()
                        burnup= [float(x) for x in parts[1:]] # burnup is in GWD/MTIHM
                        self.latburn = burnup
                        statepoint = np.linspace(1, len(burnup), len(burnup))
                        found_statepoint = False
                        index= 1
                    if ('Depletion/Decay Initial Mass Summary' in line):
                        found_startresult= True
                    elif (found_startresult==True) and ('Statepoint ' + str(index) + '\n' in line) and (index<=len(burnup)):
                        found_statepoint = True
                        print('State ' + str(index) + ' Burnup ' + str(burnup[index-1]))
                    elif (found_statepoint==True) and '|------|----------|----------|----------|----------|----------|----------|' in line:
                        found_energy= True
                        line_count= 1
                        fluxweightedER= np.zeros((2,2))
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count == 1) or (line_count == 2)):
                        parts= line.split()
                        fluxweightedER[line_count-1,0]= float(parts[3].split('|')[0])
                        line_count += 1
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count == 10) or (line_count == 11)):
                        parts= line.split()
                        print(parts)
                        fluxweightedER[line_count-11,1]= float(parts[7])
                        line_count += 1
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count == 12)):
                        line_count= 0
                        er= fluxweightedER[0,0]*fluxweightedER[0,1] + fluxweightedER[1,0]*fluxweightedER[1,1]
                        latcomp= np.append(latcomp, [burnup[index-1], er])
                        found_energy= False
                    elif (found_statepoint==True) and (found_energy==True) and ((line_count != 1) or (line_count != 2) or (line_count != 10) or (line_count != 11) or (line_count != 12)):
                        line_count += 1
                    elif (found_statepoint==True) and 'U-235 Concentration' in line:
                        parts= line.split()
                        #print('U5')
                        latcomp= np.append(latcomp,parts[3])
                    elif (found_statepoint==True) and 'U-238 Concentration' in line:
                        parts= line.split()
                        #print('U8')
                        latcomp= np.append(latcomp, parts[3])
                    elif (found_statepoint==True) and 'Pu-239 Concentration' in line:
                        parts= line.split()
                        #print('PU9')
                        latcomp= np.append(latcomp, parts[3])
                    elif (found_statepoint==True) and 'Pu-241 Concentration' in line:
                        parts= line.split()
                        #print('PU41')
                        latcomp= np.append(latcomp, parts[3])
                    elif (found_statepoint==True) and 'Neutrons Per Fission' in line:
                        #print('Nubar')
                        found_nubar= True
                        nubar= 0
                        line_count = 0
                    elif (found_statepoint==True) and (found_nubar==True) and (line_count < 1):
                        line_count += 1
                    elif (found_statepoint==True) and (found_nubar==True) and (line_count >= 1) and (line_count < line_pins_read): 
                        parts= line.split()
                        adding= [float(x) for x in parts[:]]
                        nubar= nubar + np.sum(adding)
                        line_count += 1
                    elif (found_statepoint==True) and (found_nubar==True) and (line_count == line_pins_read): 
                        parts= line.split()
                        adding= [float(x) for x in parts[:]]
                        nubar= nubar + np.sum(adding)
                        nubar= nubar/latfuelpins
                        latcomp= np.append(latcomp, nubar)
                        line_count= 0
                        found_nubar= False
                        found_statepoint= False
                        print('Lattice nubar is: ' + str(nubar) + ' for statepoint ' + str(index))
                        print('Lattice ER is: ' + str(er)+ ' for statepoint ' + str(index))
                        
                        if (self.polarisoption == 2) or (self.polarisoption == 3): # option 2 and option 3 require spectrum extraction from Polaris output

                            chi_values = self.extract_spectrum(index,filefinegroup,self.groups)
                            ene_vector= []
                            chi_vector= []

                            for ene_group, _ in chi_values:
                                ene_vector.append(ene_group)
                            ene_vector= np.array(ene_vector)

                            for _, chi_group in chi_values:
                                chi_vector.append(chi_group)
                            chi_vector= np.array(chi_vector)

                            latcomp= np.append(latcomp, np.flip(ene_vector))
                            latcomp= np.append(latcomp, np.flip(chi_vector))

                        index += 1

                self.latcomp.append((latcomp))

    # VERA related methods

    def _read_vera_card_block(self, lines: List[str], card: str) -> List[List[str]]:
        """
        Collect the token rows belonging to a VERA map card (assm_map, core_shape, ...).

        The rows of a VERA map card follow the card name on the subsequent lines and the
        block is terminated by the first blank line. Inline comments (introduced by '!')
        are stripped before tokenising.
        """
        rows = []
        found = False
        for line in lines:
            stripped = line.split('!')[0].rstrip()
            if not found:
                if stripped.strip().startswith(card):
                    found = True
                    # a card may carry its first row on the same line
                    tail = stripped.strip()[len(card):].split()
                    if tail:
                        rows.append(tail)
                continue
            if not stripped.strip():
                break
            rows.append(stripped.split())
        return rows

    def _expand_vera_map(self, rows: List[List[str]], nvera: int) -> np.ndarray:
        """
        Expand a VERA map given as an octant, a quarter or a full map into a full nvera x nvera map.

        VERA maps are written outwards from the core centre. An octant is stored as a triangular
        block (row r holds at most r+1 entries) and is mirrored across the diagonal to recover the
        quarter; the quarter is then mirrored about the centre row and column to recover the full
        core. Entries falling outside the core shape are truncated by the caller.
        """
        centre = nvera // 2
        half = centre + 1

        # classify the layout from the row lengths
        if len(rows) == nvera and all(len(r) == nvera for r in rows):
            layout = 'full'
        elif all(len(r) <= i + 1 for i, r in enumerate(rows)):
            layout = 'octant'
        else:
            layout = 'quarter'
        print(f'VERA map layout recognised as: {layout}')

        if layout == 'full':
            return np.array([[self._vera_label_to_type(v) for v in r] for r in rows], dtype=int)

        quarter = np.zeros((half, half), dtype=int)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                quarter[r, c] = self._vera_label_to_type(val)

        if layout == 'octant':
            # mirror across the diagonal to complete the quarter
            for r in range(half):
                for c in range(half):
                    if quarter[r, c] == 0 and quarter[c, r] != 0:
                        quarter[r, c] = quarter[c, r]

        # mirror the quarter about the centre row and column
        full = np.zeros((nvera, nvera), dtype=int)
        for i in range(nvera):
            for j in range(nvera):
                full[i, j] = quarter[abs(i - centre), abs(j - centre)]
        return full

    def _vera_label_to_type(self, label: str) -> int:
        """
        Convert a VERA assembly label to the integer assembly type used by the PARCS structures.

        Numeric labels are taken as-is; non-numeric labels ('-', 'A', ...) are mapped to 0 when
        they denote an empty position and to a stable sequential index otherwise.
        """
        label = label.strip()
        if label in ('-', '', '0'):
            return 0
        if label.lstrip('-').isdigit():
            return int(label)
        # non numeric label: assign a stable id based on first appearance
        if not hasattr(self, '_vera_label_ids'):
            self._vera_label_ids = {}
        if label not in self._vera_label_ids:
            self._vera_label_ids[label] = len(self._vera_label_ids) + 1
        return self._vera_label_ids[label]

    def extract_assyradial_VERA(
            self, geom: "Geometry", filepath: Union[str, Path]) -> None:
        """
        Extracts assembly radial configuration from a VERA input file (assm_map card).

        Produces the same structure as extract_assyradial: a list of (x_index, y_index, value)
        tuples covering the full geom.nass x geom.nass map, where value is the assembly type and
        0 marks a position without a fuel assembly.

        The assm_map card may be given as an octant, a quarter or a full map; the layout is
        recognised automatically and expanded to the full core. The VERA map covers only the
        fuel region (core_shape), whereas the PARCS/neuthos map carries an additional reflector
        ring, so the VERA map is inserted with an offset of (geom.nass - nvera) // 2.

        Parameters
        ----------
        geom : Geometry
            Geometry object providing the neuthos core map size (nass).
        filepath : Union[str, Path]
            Path to the VERA input file (.inp).

        Returns
        -------
        None
            Populates self.assyradial.
        """

        print('Extracting assembly radial configuration from VERA input ...')

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # core size: prefer the core_shape block, fall back to the 'size' card
        shape_rows = self._read_vera_card_block(lines, 'core_shape')
        if shape_rows:
            core_shape = np.array([[int(v) for v in r] for r in shape_rows], dtype=int)
            nvera = core_shape.shape[0]
        else:
            core_shape = None
            nvera = None
            for line in lines:
                parts = line.split('!')[0].split()
                if len(parts) >= 2 and parts[0] == 'size':
                    nvera = int(parts[1])
                    break
            if nvera is None:
                print('No core_shape or size card found in the VERA input.')
                raise SystemExit

        map_rows = self._read_vera_card_block(lines, 'assm_map')
        if not map_rows:
            print('No assm_map card found in the VERA input.')
            raise SystemExit

        full = self._expand_vera_map(map_rows, nvera)

        # positions outside the core shape carry no assembly
        if core_shape is not None:
            full = full * core_shape

        # the neuthos/PARCS map carries a reflector ring around the VERA fuel map
        offset = (geom.nass - nvera) // 2
        if offset < 0:
            print(f'geom.nass ({geom.nass}) is smaller than the VERA core size ({nvera}).')
            raise SystemExit
        print(f'VERA core size {nvera} inserted in a {geom.nass} map with offset {offset}')

        for x_index in range(1, geom.nass + 1):
            for y_index in range(1, geom.nass + 1):
                i = x_index - 1 - offset
                j = y_index - 1 - offset
                if 0 <= i < nvera and 0 <= j < nvera:
                    value = int(full[i, j])
                else:
                    value = 0
                self.assyradial.append((x_index, y_index, value))

    def extract_assyaxial_VERA(
            self, geom: "Geometry", filepath: Union[str, Path]) -> None:
        """
        Extracts assembly axial configuration from a VERA input file (axial cards).

        Produces the same structure as extract_assyaxial: a list of
        ['FUEL'|'REFL', assy_type, assy_geom] entries, where assy_geom holds one lattice id per
        axial node ordered from top to bottom (the ordering used by the PARCS burnup profiles).

        Only the axial cards of the [ASSEMBLY] section are considered, so that the axial cards of
        [INSERT] (control rods, burnable poisons) are not mistaken for assembly types. An axial
        card reads 'axial <type> <z0> <label1> <z1> <label2> <z2> ...'; a segment whose label
        starts with 'LAT' is a fuel segment and its lattice id is the numeric part of the label
        (LAT18 -> 18). Nodes outside every fuel segment are axial reflector and take the id 0.

        Parameters
        ----------
        geom : Geometry
            Geometry object providing the axial node mesh (z_core) and the total node count
            (naxial). If geom.z_core is empty the axial_edit_bounds card of the input is used.
        filepath : Union[str, Path]
            Path to the VERA input file (.inp).

        Returns
        -------
        None
            Populates self.assyaxial, and self.fuellat / self.refllat when these are still empty.
        """

        print('Extracting assembly axial configuration from VERA input ...')

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # axial node boundaries: prefer the mesh already used by the Geometry object
        if len(geom.z_core) > 1:
            bounds = [float(z) for z in geom.z_core]
            print('Using the axial mesh already stored in geom.z_core')
        else:
            bounds = []
            for line in lines:
                parts = line.split('!')[0].split()
                if parts and parts[0] == 'axial_edit_bounds':
                    bounds = [float(x) for x in parts[1:]]
                    break
            if not bounds:
                print('No axial mesh available: populate geom.z_core or provide an axial_edit_bounds card.')
                raise SystemExit
            print('Using the axial_edit_bounds card of the VERA input')

        nactive = len(bounds) - 1

        # collect the axial cards of the [ASSEMBLY] section only
        section = None
        cards = []
        for line in lines:
            stripped = line.split('!')[0].strip()
            if stripped.startswith('['):
                section = stripped.strip('[]').upper()
                continue
            parts = stripped.split()
            if section == 'ASSEMBLY' and parts and parts[0] == 'axial':
                cards.append(parts)

        if not cards:
            print('No axial cards found in the [ASSEMBLY] section of the VERA input.')
            raise SystemExit

        fuel_ids = set()
        for parts in cards:
            assy_type = self._vera_label_to_type(parts[1])

            # the card alternates elevation / label / elevation / label / ... / elevation
            tokens = parts[2:]
            segments = []
            for idx in range(1, len(tokens) - 1, 2):
                label = tokens[idx]
                z_lo = float(tokens[idx - 1])
                z_hi = float(tokens[idx + 1])
                segments.append((z_lo, z_hi, label))

            # assign a lattice id to every active axial node
            assy_geom = []
            for k in range(nactive):
                centre = 0.5 * (bounds[k] + bounds[k + 1])
                lattice = 0
                for z_lo, z_hi, label in segments:
                    if z_lo <= centre <= z_hi and label.upper().startswith('LAT'):
                        digits = re.findall(r'\d+', label)
                        lattice = int(digits[0]) if digits else self._vera_label_to_type(label)
                        break
                assy_geom.append(lattice)

            # pad with axial reflector nodes so that the profile spans geom.naxial nodes
            pad = geom.naxial - nactive
            if pad < 0:
                print(f'geom.naxial ({geom.naxial}) is smaller than the number of active nodes ({nactive}).')
                raise SystemExit
            bottom = pad // 2
            top = pad - bottom
            assy_geom = [0] * bottom + assy_geom + [0] * top

            fuel_ids.update(l for l in assy_geom if l != 0)
            kind = 'FUEL' if any(l != 0 for l in assy_geom) else 'REFL'
            # NOTE: reverse the list to have the lattice from top to bottom, REASON: burnup profiles are provided from top to bottom
            self.assyaxial.append([kind, assy_type, assy_geom[::-1]])
            print(f'Assembly type {assy_type} ({kind}): {nactive} active nodes, {pad} reflector nodes')

        # the downstream source builders classify each node through fuellat / refllat: fill them
        # in from the VERA cards when the PARCS lattice tables have not been read
        if not self.fuellat and not self.refllat:
            self.fuellat = sorted(fuel_ids)
            self.refllat = [0]
            print(f'fuellat / refllat populated from the VERA input: fuellat={self.fuellat}, refllat={self.refllat}')
        else:
            print('fuellat / refllat already populated: leaving the existing lattice tables untouched')

    def extract_pinpowerVERA_2D(
            self, geom: "Geometry", out: "Outputs", filepath: Union[str, Path], step: int) -> None:
        """
            Extracts 2D pinpower from a VERA (MPACT) .h5 output file.

            Parameters
            ----------
            geom : Geometry
                Geometry object providing the VERA pin coordinates (coordinatesVERA_with_index),
                the number of fuel pins, the pin radius and the active height.
            out : Outputs
                Outputs object for managing output directories (sanity checks).
            filepath : Union[str, Path]
                Path to the VERA .h5 output file containing the pin power datasets.
            step : int, optional
                Burnup state point to extract, counted from 0. Default is 0.

            Returns
            -------
            None
                Populates self.pinpowerdataVERA2D with the extracted pin power data.

            Notes
            -----
            - Pin powers are assumed to be at full power (powlevel = 1).
            - The output power is expressed in W/cm (linear power, no axial mesh weighting).
            - If self.explicitdeploption is True, the pin-wise U-235, U-238, Pu-239 and Pu-241
              number densities are read from the same state point (axial slice 0) and appended
              to each tuple, matching the layout produced by extract_pinpowerVERA.
        """

        print('The average power density is:')
        self.corevol = geom.nfuelpins * math.pi * (geom.pin_radius)**2 * (geom.active_height)        # cm3
        self.avgpowdens= self.asspower / (self.corevol)                                              # W/cm3
        print(self.avgpowdens)                                                                       # W/cm3

        print('Step 2.0: extracting pin power level ...')
        with h5py.File(filepath, 'r') as h5file:
            pin_powers = h5file[f'STATE_{step+1:04d}/pin_powers'][:]
            if self.explicitdeploption:
                u235 = h5file[f'STATE_{step+1:04d}/pin_isotopes_U-235'][:]
                u238 = h5file[f'STATE_{step+1:04d}/pin_isotopes_U-238'][:]
                pu239 = h5file[f'STATE_{step+1:04d}/pin_isotopes_Pu-239'][:]
                pu241 = h5file[f'STATE_{step+1:04d}/pin_isotopes_Pu-241'][:]
            case_number= step # steps are counted from 0
            k= 0 #2D case
            powlevel= 1 # FULL POWER ASSUMPTION
            for coord_info in geom.coordinatesVERA_with_index:
                # burnup statepoint, VERA unique assembly index, assembly row index, assembly column index, assembly z index, pin row index, pin column index, pin x-coordinate, pin y-coordinate, power
                if pin_powers[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]] != 0: # check if the pin power is not zero
                    print(f'... Extracting VERA pin power data for assembly {coord_info[6]} ...')
                    # NOTE 1: the output will be in W/cm | ALTERNATIVE using node volume considering the whole assembly geom.nodevolume[k-1]/225
                    # NOTE 2: added a field for U-235, U-238, Pu-239, Pu-241 (2D uses axial slice 0)
                    if self.explicitdeploption:
                        self.pinpowerdataVERA2D.append((case_number, coord_info[6], coord_info[0], coord_info[1], k, coord_info[2], coord_info[3], coord_info[4], coord_info[5],
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]]*self.avgpowdens*math.pi*(geom.pin_radius)**2*powlevel,
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]], u235[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]],
                                                 u238[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]], pu239[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]],
                                                 pu241[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]]))
                    else :
                        self.pinpowerdataVERA2D.append((case_number, coord_info[6], coord_info[0], coord_info[1], k, coord_info[2], coord_info[3], coord_info[4], coord_info[5],
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]]*self.avgpowdens*math.pi*(geom.pin_radius)**2*powlevel,
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, 0, coord_info[6]]))

        # SANITY CHECK: total power (check for one burnup step)
        outpath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        total_power= 0
        count = 0
        for power in self.pinpowerdataVERA2D:
            if (power[0] == step):
                count += 1
                total_power += power[9]

        with open(outpath / f'sanity_check_VERA2Dpin_{step}.txt', 'w') as f:
            f.write(f'Average linear power: {total_power/count if count else 0.0}\n')
            f.write(f'Count: {count}')

    def extract_pinpowerVERA(
            self, geom: "Geometry", out: "Outputs", filepath: Union[str, Path], step: int) -> None: 
        """
            Extracts 3D pinpower from a VERA (MPACT) .h5 output file.

            Parameters
            ----------
            geom : Geometry
                Geometry object providing the VERA pin coordinates (coordinatesVERA_with_index),
                the axial mesh heights, the number of fuel pins, the pin radius and the active height.
            out : Outputs
                Outputs object for managing output directories (sanity checks).
            filepath : Union[str, Path]
                Path to the VERA .h5 output file containing the pin power datasets.
            step : int, optional
                Burnup state point to extract, counted from 0. Default is 0. The corresponding
                VERA state is STATE_{step+1:04d}, as VERA numbers its state points from 1.

            Returns
            -------
            None
                Populates self.pinpowerdataVERA with the extracted pin power data.

            Notes
            -----
            - Pin powers are assumed to be at full power (powlevel = 1).
            - The output power is weighted by the axial mesh height, so it is expressed in W.
            - If self.explicitdeploption is True, the pin-wise U-235, U-238, Pu-239 and Pu-241
              number densities are read from the same state point and appended to each tuple.
        """

        print('The average power density is:')
        self.corevol = geom.nfuelpins * math.pi * (geom.pin_radius)**2 * (geom.active_height)        # cm3
        self.avgpowdens= self.asspower / (self.corevol)                                              # W/cm3
        print(self.avgpowdens)                                                                       # W/cm3

        print('Step 2.0: extracting pin power level ...')
        with h5py.File(filepath, 'r') as h5file:
            pin_powers = h5file[f'STATE_{step+1:04d}/pin_powers'][:]
            if self.explicitdeploption:
                u235 = h5file[f'STATE_{step+1:04d}/pin_isotopes_U-235'][:]
                u238 = h5file[f'STATE_{step+1:04d}/pin_isotopes_U-238'][:]
                pu239 = h5file[f'STATE_{step+1:04d}/pin_isotopes_Pu-239'][:]
                pu241 = h5file[f'STATE_{step+1:04d}/pin_isotopes_Pu-241'][:]
            case_number= step # case numbers are counted from zero
            powlevel= 1 # FULL POWER ASSUMPTION
            for coord_info in geom.coordinatesVERA_with_index:
                # burnup statepoint, VERA unique assembly index, assembly row index, assembly column index, assembly z index, pin row index, pin column index, pin x-coordinate, pin y-coordinate, power
                if pin_powers[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]] != 0: # check if the pin power is not zero
                    print(f'... Extracting VERA pin power data for assembly {coord_info[6]} ...')
                    # NOTE 1: the output will be in W/cm | ALTERNATIVE using node volume considering the whole assembly geom.nodevolume[k-1]/225
                    # NOTE 2: added a filed for U-235, U-238, Pu-239, Pu-241
                    if self.explicitdeploption:
                        self.pinpowerdataVERA.append((case_number, coord_info[6], coord_info[0], coord_info[1], coord_info[8], coord_info[2], coord_info[3], coord_info[4], coord_info[5],
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]]*self.avgpowdens*math.pi*(geom.pin_radius)**2*powlevel*geom.meshheight[coord_info[8]],
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]], u235[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]],
                                                 u238[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]], pu239[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]],
                                                 pu241[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]]))
                    else :
                        self.pinpowerdataVERA.append((case_number, coord_info[6], coord_info[0], coord_info[1], coord_info[8], coord_info[2], coord_info[3], coord_info[4], coord_info[5],
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]]*self.avgpowdens*math.pi*(geom.pin_radius)**2*powlevel*geom.meshheight[coord_info[8]],
                                                 pin_powers[coord_info[2]-1, coord_info[3]-1, coord_info[8], coord_info[6]]))

        # Sort the pin power data based on i, j, k (to keep the same order as PARCS for readability)
        self.pinpowerdataVERA.sort(key=lambda x: (x[2], x[3], x[4]))

        # SANITY CHECK: total power (check for one burnup step)
        outpath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        total_power= 0
        count = 0
        for power in self.pinpowerdataVERA:
            if (power[0] == step):
                count += 1
                total_power += power[9]

        with open(outpath / f'sanity_check_VERA3Dpin_{step}.txt', 'w') as f:
            f.write(f'Average pin power: {total_power/count if count else 0.0}\n')
            f.write(f'Count: {count}')