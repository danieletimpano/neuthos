# src/neuthos/outputs.py

from dataclasses import dataclass, field
from fileinput import filename
from pathlib import Path
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import csv 
import os

# Create an Outputs dataclass to handle output-related functionalities
@dataclass
class Outputs:
    """
    Handles output and visualization functionalities for neutronics calculations.
    
    This class provides methods to write various data outputs to CSV files and
    create visualizations for source definitions. It supports both 2D and 3D
    visualizations and manages output routing through a configurable base directory.
    
    :param base_dir: Base directory for routing all output files. Defaults to current directory.
    :type base_dir: Path
    """
    base_dir: Path = Path(".")  # optional; can be used to route outputs

    def write_PARCS_pin_coordinates(
        self,
        geom: "Geometry",
        outputpath: Union[str, Path],
    ) -> Path:
        """
        Write PARCS pin-level coordinates from a Geometry instance to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per pin with indices and
        Cartesian coordinates.
        """
        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        outputpath.parent.mkdir(parents=True, exist_ok=True)

        with outputpath.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["i", "j", "m", "n", "x_pin", "y_pin"])
            for (i, j, m, n, x_pin, y_pin) in geom.coordinates:
                writer.writerow([i, j, m, n, x_pin, y_pin])

        return outputpath
    
    def write_VERA_pin_coordinates(
        self,
        geom: "Geometry",
        outputpath: Union[str, Path],
    ) -> Path:
        """
        Write VERA pin-level coordinates with VERA indices from a Geometry instance to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per pin with indices, VERA index, and
        Cartesian coordinates.
        """
        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        outputpath.parent.mkdir(parents=True, exist_ok=True)

        with outputpath.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["i", "j", "m", "n", "x_pin", "y_pin", "VERA_index", "Axial_height", "Axial_node_index_k"])
            print('Warning: VERA pin coordinates information may not include axial information if 2D option was selected.')
            for row in geom.coordinatesVERA_with_index:
                writer.writerow(row)

        return outputpath
    
    def write_assyaxial(
        self, cycle: "Cycle", outputpath: Union[str, Path]) -> Path:
        """
        Write assembly axial configuration to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per assembly type with its axial configuration.
        """
        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        print('Saving assembly axial configuration ...')

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['F/R', 'assy_type', 'assy_geom (from top to bottom)'])
            for assy in cycle.assyaxial:
                writer.writerow(assy)
        
        return outputpath

    def write_assyradial(
        self, cycle: "Cycle", outputpath: Union[str, Path]) -> Path:
        """
        Write assembly radial configuration to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per assembly with its radial configuration.
        """
        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        print('Saving assembly radial configuration ...')

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x_index', 'y_index', 'assy_type'])
            for assy in cycle.assyradial:
                writer.writerow(assy)
        
        return outputpath

    def write_exposureindex_to_csv(
            self, cycle: "Cycle", outputpath: Union[str, Path]) -> Path:
        """
        Write exposure matrix to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per assembly with its radial configuration.
        """

        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        print('Saving assembly exposure data ...')

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x_index', 'y_index', 'exp_index'])
            for assy in cycle.assyexp:
                writer.writerow(assy) 
        
        return outputpath

    def write_latcomp_to_csv(
            self, cycle: "Cycle", outputpath: Union[str, Path]) -> Path:
        
        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        print('Saving lattice material composition ...')

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['latID', 'bu-1', 'U-235', 'U-238', 'Pu-239', 'Pu-241', '...'])
            for comp in cycle.latcomp:
                writer.writerow(comp) 

        return outputpath

    def write_lattype_to_csv(
            self, cycle: "Cycle", outputpath: Union[str, Path]) -> Path:

        """
        Write lattice type data to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per lattice type with its properties.
        """

        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        print('Saving lattice type data ...')

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['latID', 'lattype', 'f_refl', 'enrichment', 'bp', 'sg', 'reference'])
            for type in cycle.lattype:
                writer.writerow(type) 

        return outputpath

    def write_asspower_to_csv(            
            self, cycle: "Cycle", outputpath: Union[str, Path]) -> Path:
        
        """
        Write assembly power data to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per assembly with its power data.
        """
        
        print('Saving core power to .csv ...')

        outputpath = Path(outputpath)
        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time_info', 'assemly_x', 'assembly_y', 'axial_plane', 'ass_power'])
            for power in cycle.asspowerdata:
                writer.writerow(power)

        return outputpath

    def write_pinpower_to_csv(
            self, cycle: "Cycle", outputpath: Union[str, Path], flag: str = '3D') -> Path:
        """
        Write pin power data to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per pin with its power data.
        """
        
        print('Saving pin power information to .csv ...')

        outputpath = Path(outputpath)
        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['case_number', 'i', 'j', 'k', 'x_index', 'y_index', 'node_power'])
            if flag == '2D':
                for power in cycle.pinpowerdata2D:
                    writer.writerow(power)
            elif flag == '3D':
                for power in cycle.pinpowerdata:
                    writer.writerow(power)
        
        return outputpath
    
    def write_pinpowerVERA_to_csv(
            self, cycle: "Cycle", outputpath: Union[str, Path], flag: str = '3D') -> Path:
        """
        Write pin power data to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains one row per pin with its power data.
        """
        
        print('Saving pin power information to .csv ...')

        outputpath = Path(outputpath)
        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['case_number', 'vera_index', 'assembly_row', 'assembly_column', 'assembly_z', 'pin_row', 'pin_column', 'pin_x', 'pin_y', 'node_power', 'pin_power_factor'])
            if flag == '2D':
                for power in cycle.pinpowerdataVERA2D:
                    writer.writerow(power)
            elif flag == '3D':
                for power in cycle.pinpowerdataVERA:
                    writer.writerow(power)

        return outputpath
            
    def write_asssourceinfo_to_csv(
            self, source: "Source", outputpath: Union[str, Path]) -> Path:
        
        """
        Write assembly source information to a CSV file.
        The output file is created at the given path (relative to the output base directory, if configured) 
        and contains one row per assembly with its source information.

        """
        
        print('Saving assembly source information to .csv ...')

        outputpath = Path(outputpath)
        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['case_number', 'i', 'j', 'k','node_power','nsource', 'spectrum', 'fuel_type'])
            for source in source.sourceassy:
                writer.writerow(source)
            
        return outputpath

    def write_pinsourceinfo_to_csv(
            self, source: "Source", outputpath: Union[str, Path], flag: str = '3D') -> Path:
        
        """
        Write pin source information to a CSV file.
        The output file is created at the given path (relative to the output base directory, if configured) 
        and contains one row per pin with its source information.
        """
        
        print('Saving pin source information to .csv ...')

        outputpath = Path(outputpath)
        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        with open(outputpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['case_number', 'i', 'j', 'k', 'x_index', 'y_index', 'node_power','nsource'])
            if flag == '2D':
                if source.sourcepin2D: # implicit treatment of the code selection
                    for source in source.sourcepin2D:
                        writer.writerow(source)
                else:
                    for source in source.sourcepinVERA2D:
                        writer.writerow(source)
            elif flag == '3D':
                if source.sourcepin: # implicit treatment of the code selection
                    for source in source.sourcepin:
                        writer.writerow(source)
                else:
                    for source in source.sourcepinVERA:
                        writer.writerow(source)

        return outputpath

    def visualize_source_ass(
            self, geom: "Geometry", source: "Source", flag: str) -> Path:
        """
        Visualize assembly source definition.
        The visualization can be either 2D (at core midplane) or 3D based on the flag provided.
        """
        
        print('Visualizing assembly source definition ...')

        outpath = Path(os.path.join(self.base_dir, 'plots'))
        outpath.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
        massimo = 0
        sums= 0
        time = source.step #read required timestep for plotting
        
        if flag== '2D': # if flag is equal to 2D you may plot at core midplane

            # Create a matrix to store the source values
            source_matrix = np.zeros((geom.nass, geom.nass))

            for ass in source.sourceassy:
                if ass[3] == geom.naxial // 2:  # Only for mid-core plane in 2D
                    time = ass[0]
                    x_core = ass[1] - 1
                    y_core = ass[2] - 1
                    source_matrix[x_core, y_core] = ass[5]
                    massimo += ass[5]
                    sums += 1

            # Substitute remaining zeros with nan and plot nan as white values
            source_matrix[source_matrix == 0] = np.nan
            avg= massimo/sums

            cax = ax.imshow(source_matrix/avg, cmap='jet', vmin=0.5, vmax=1.2)
            ax.set_xticks(np.arange(0, geom.nass))
            ax.set_yticks(np.arange(0, geom.nass))
            ax.set_xticklabels(np.arange(1, geom.nass + 1))
            ax.set_yticklabels(np.arange(1, geom.nass + 1))
            fig.colorbar(cax, label='Normalized Source Term')
            # Annotate the graph with source_matrix/massimo corresponding values
            for i in range(geom.nass):
                for j in range(geom.nass):
                    if not np.isnan(source_matrix[i, j]):
                        ax.text(j, i, f'{source_matrix[i, j]/avg :.2f}', ha='center', va='center', color='black', fontsize=6)

            # Add checkerboard lines
            for i in range(geom.nass + 1):
                ax.axhline(i - 0.5, color='black', linewidth=0.5)
                ax.axvline(i - 0.5, color='black', linewidth=0.5)

            ax.set_title('Assembly Source Definition at mid-core height')
            plt.savefig(outpath / ('assembly_source_definition_t' + str(time) + '_2D.png'), bbox_inches= 'tight')
        
        elif flag== '3D':
            #save colormap data
            temp= []
            #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
            fig = plt.figure(dpi=300, figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            for ass in source.sourceassy:
                    time = ass[0]
                    x_core = ass[1] - 1
                    y_core = ass[2] - 1
                    z_core= ass[3]
                    temp.append(ass[5]/4e+16)
                # the values of the burnup profile are represented by the colormap
                    ax.bar3d(x_core, y_core, z_core, 1, 1, 1, shade=True, color=plt.cm.bwr(ass[5]/4e+16), edgecolor= 'black', linewidth= 0.2)
                    # Add the legend to the plot
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            plt.grid()
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr, norm=plt.Normalize(vmin=np.min(temp)/4e+16, vmax=1)), ax=ax, label='Normalized Source Strength (-)')
            #change the orientation of the graph
            ax.view_init(elev=30, azim=60)
            plt.savefig(outpath / ('assembly_source_definition_t' + str(time) + '_3D.png'), bbox_inches= 'tight')
        
        return outpath

    def visualize_source_pin(
            self, geom: "Geometry", source: "Source", flag: str) -> Path:
        """
        Visualize pin source definition.
        The visualization can be either 2D (at core midplane) or 3D based on the flag provided.
        If 3D is selected, a .csv file readable by Paraview Table to Points option is created.
        Use Paraview to visualize the source for 3D definition.
        """
        
        print('Visualizing assembly source definition ...')

        outpath = Path(os.path.join(self.base_dir, 'plots'))
        outpath.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
        massimo = 0
        sums = 0
        time = source.step

        if flag=='2D': # plot core midplane
            # Create a matrix to store the source values
            source_matrix = np.zeros((geom.nass * geom.npin, geom.nass * geom.npin))

            # implicit treatment of the code selection - check which one between sourcepin and sourcepin2D is not empty and use it for plotting
            if source.sourcepin2D:
                source_list = source.sourcepin2D
            elif source.sourcepinVERA2D:
                source_list = source.sourcepinVERA2D
            elif source.sourcepin:
                source_list = source.sourcepin
            else:
                source_list = source.sourcepinVERA

            for pin in source_list:
                if pin[3] == geom.naxial // 2:  # Only for midplane in 2D
                    time = pin[0]
                    x_core = (pin[1] - 1) * geom.npin + (pin[4] - 1)
                    y_core = (pin[2] - 1) * geom.npin + (pin[5] - 1)
                    source_matrix[x_core, y_core] = pin[7]
                    massimo += pin[7]
                    sums += 1

            # Substitute remaining zeros with nan and plot nan as white values
            source_matrix[source_matrix == 0] = np.nan
            if sums != 0:
                avg = massimo / sums
            else: 
                print("Warning: No source data found for the specified mid-core plane. Do not trust the produced plot.")

            cax = ax.imshow(source_matrix / avg, cmap='jet', vmin=0.5, vmax=1.2)
            ax.set_xticks(np.arange(0, geom.nass * geom.npin, geom.npin))
            ax.set_yticks(np.arange(0, geom.nass * geom.npin, geom.npin))
            ax.set_xticklabels(np.arange(1, geom.nass + 1))
            ax.set_yticklabels(np.arange(1, geom.nass + 1))
            fig.colorbar(cax, label='Normalized Source Term')

            # Add checkerboard lines
            for i in range(geom.nass + 1):
                ax.axhline(i * geom.npin - 0.5, color='black', linewidth=0.5)
                ax.axvline(i * geom.npin - 0.5, color='black', linewidth=0.5)

            # Add subpixel lines
            for i in range(geom.nass * geom.npin + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.2, linestyle='--')
                ax.axvline(i - 0.5, color='gray', linewidth=0.2, linestyle='--')

            ax.set_title('Pin Source Definition at mid-core height')
            plt.savefig(outpath / ('pin_source_definition_t' + str(time) + '.png'), bbox_inches='tight')

        elif flag== '3D':
            
            # implicit treatment of the code choice
            if source.sourcepin:
                source_list = source.sourcepin
            else:
                source_list = source.sourcepinVERA

            #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
            fig = plt.figure(dpi=300, figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            print('Selected Paraview Option ...')

            with open(outpath / f'pin_source_{time}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for pin in source.sourcepin:
                    time = pin[0]
                    x_core = (pin[1] - 1) * geom.npin + (pin[4] - 1)
                    y_core = (pin[2] - 1) * geom.npin + (pin[5] - 1)
                    z_core= pin[3]
                    writer.writerow([x_core, y_core, z_core, pin[7]])

        return outpath

    def write_custom_csv(
            self, data: List[Tuple], outputpath: Union[str, Path]) -> Path:
        """
        Write custom output data to a CSV file.

        The output file is created at the given path (relative to the output base
        directory, if configured) and contains the provided header and data rows.
        """
        outputpath = Path(outputpath)

        # Route relative paths under base_dir (so that outputs can be grouped)
        if not outputpath.is_absolute():
            outputpath = self.base_dir / outputpath

        outputpath.parent.mkdir(parents=True, exist_ok=True)

        with outputpath.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)
        return outputpath