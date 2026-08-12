# src/neuthos/source.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union
from neuthos.cycle import Cycle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import csv
import os
import re

@dataclass
class Source:
    """
    Manages neutron source term generation for LWR shielding calculations.
    
    This class handles the computation and output of neutron source terms at both assembly
    and pin levels, including fission spectrum calculations, power-to-source conversions,
    and truncation options for symmetry geometries.
    
    Attributes
    ----------
    step : int
        Step number used for source term extraction from cycle data.
    bins : np.ndarray
        Energy bin edges defining the fission spectrum discretization (linspace array).
    truncoption : bool
        Flag to control spectrum truncation at upper energy bound (False: no truncation, 
        True: truncation enabled).
    assytype_to_mcmaterial : dict
        Mapping dictionary linking Serpent material names to PARCS assembly type identifiers.
    sourceoption : str, optional
        Type of source definition ('pointsource' or 'volumesource'). Default is 'pointsource'.
    trunc_ass_X : List[Tuple[int, int]], optional
        Assembly coordinates along X truncation symmetry line.
    trunc_ass_Y : List[Tuple[int, int]], optional
        Assembly coordinates along Y truncation symmetry line.
    trunc_pin_X : List[Tuple[int, int]], optional
        Pin grid indices along X truncation symmetry line.
    trunc_pin_Y : List[Tuple[int, int]], optional
        Pin grid indices along Y truncation symmetry line.
    sourceassy : List[Tuple], optional
        Computed assembly-level source data. Each tuple contains:
        (step, i_coord, j_coord, k_coord, power, source_term, spectrum, fuel_type, energy_bounds).
    sourcepin : List[Tuple], optional
        Computed pin-level source data. Each tuple contains:
        (step, i_coord, j_coord, k_coord, pin_x, pin_y, power, source_term, spectrum, fuel_type).
    sourcepin2D : List[Tuple], optional
        Computed 2D pin-level source data. Each tuple contains:
        (step, i_coord, j_coord, x_index, y_index, source_term).
    """
    # --- source inputs ---
    step: int
    bins: np.ndarray
    truncoption: bool
    assytype_to_mcmaterial: dict
    sourceoption: str = 'pointsource'
    trunc_ass_X: List[Tuple[int, int]] = None
    trunc_ass_Y: List[Tuple[int, int]] = None
    trunc_pin_X: List[Tuple[int, int]] = None
    trunc_pin_Y: List[Tuple[int, int]] = None
    trunc_pinsym_X: List[Tuple[int, int]] = None
    trunc_pinsym_Y: List[Tuple[int, int]] = None

    # --- computed / parsed outputs ---
    sourceassy : List[Tuple[int, int, int]] = field(default_factory=list)
    sourcepin : List[Tuple[int, int, int, int, int, int, float]] = field(default_factory=list)
    sourcepin2D : List[Tuple[int, int, int, int, int, float]] = field(default_factory=list)
    sourcepinVERA : List[Tuple] = field(default_factory=list)                                # VERA 3D pin source data (step, i, j, k, m, n, power, source, chi, fueltype)
    sourcepinVERA2D : List[Tuple] = field(default_factory=list)                              # VERA 2D pin source data (step, i, j, k, m, n, power, source, chi, fueltype)
    F : List[dict] = field(default_factory=list)                                             # assembly_list structure retained after a build_source_* call

    # PARCS related methods

    def build_source_ass(
        self, cycle: "Cycle", geom: "Geometry", out: "Outputs") -> None:
        """
        Build assembly-level neutron source terms from cycle and geometry data.
        
        This method processes assembly-wise burnup profiles, compositions, and generates
        neutron multiplication factors and fission spectra. It computes the conversion
        factors from power to source terms and applies truncation options if specified.
        
        Parameters
        ----------
        cycle : Cycle
            Cycle object containing assembly power data, exposure profiles, and lattice compositions.
        geom : Geometry
            Geometry object containing assembly and core mesh information.
        out : Outputs
            Outputs object for managing output directories and file paths.
        
        Returns
        -------
        None
            Populates self.sourceassy with computed assembly-level source data.
        """
        print('Preparing data for source term ...')
        
        # for step in steps:

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(geom.naxial,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[], 'eubound':[]}
            assembly_list.append(assembly)

        # define which assemblies are fuel and which are reflector or dummies
        for assembly in assembly_list:
            if assembly['assytype'] == 0:
                assembly['r-f-d']= 'DUMMY'
            elif assembly['assytype'] != 0:
                for axial in cycle.assyaxial:
                    if axial[1] == assembly['assytype']:
                        assembly['r-f-d']= axial[0]

        # extract the burnup ID from exposure info
        for assembly in assembly_list:
            for assy in cycle.assyexp:
                if assembly['coordinates'] == [assy[0], assy[1]]:
                    assembly['burnupID']= assy[2]

        # extract the burnup profile from selected exposure point for each assembly
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                assembly['buprofile'] = cycle.exposure[:, assembly['burnupID'] - 1, self.step].flatten().tolist() #STEP CONTROL - USER INPUT: currenly extracting specific burnup step: assemblies which have burnupID = 0 will have a burnup profile of 0

        # extract the lattice configuration of each assembly type
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                for axialinfo in cycle.assyaxial:
                    if axialinfo[1] == assembly['assytype']:
                        assembly['latID']= axialinfo[2]
            elif assembly['burnupID'] == []: #preassign values to reflector assembly types
                assembly['latID']= [0]*geom.naxial
                assembly['burnupID']= 0
                assembly['buprofile']= [0]*geom.naxial
                assembly['bu_clos']= [0]*geom.naxial
                assembly['nubar']= 0
                assembly['ERC']= 0
                assembly['U235']= [0]*geom.naxial
                assembly['U238']= [0]*geom.naxial
                assembly['Pu239']= [0]*geom.naxial
                assembly['Pu241']= [0]*geom.naxial
                assembly['chi']= [0]*100
        
        # extract the specific lattice composition: find the corresponding burnup point in lattice library and the composition
        for assembly in assembly_list:
            bp_index= -1 
            for lattice in assembly['latID']:
                #print('Problematic' + str(lattice))
                if (lattice in cycle.fuellat):
                    bp_index += 1
                    for latcomp in cycle.latcomp:
                        if (lattice == int(float(latcomp[0]))): # only fuel lattices should look for uranium and plutonium composition
                            # print(latcomp)
                            bppoint = assembly['buprofile'][bp_index]
                            closest_burnup = min(cycle.latburn, key=lambda x: abs(x - bppoint))
                            # print("The closest burnup is" + str(closest_burnup))
                            assembly['bu_clos'].append(closest_burnup)

                            if (cycle.polarisoption == 0) or (cycle.polarisoption == 1):
                                for idx in range(0, len(latcomp), 7):
                                    if float(latcomp[idx + 1]) == closest_burnup:
                                        assembly['ERC'].append(float(latcomp[idx + 2]))
                                        assembly['U235'].append(float(latcomp[idx + 3]))
                                        assembly['U238'].append(float(latcomp[idx + 4]))
                                        assembly['Pu239'].append(float(latcomp[idx + 5]))
                                        assembly['Pu241'].append(float(latcomp[idx + 6]))
                                        assembly['nubar'].append(float(latcomp[idx + 7]))
                                        break

                            if (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                                for idx in range(0,len(latcomp), (7+(cycle.groups-1)*2)):
                                    if float(latcomp[idx + 1]) == closest_burnup:
                                        #print('The closest lattice burnup in the library is' + str(closest_burnup))
                                        assembly['ERC'].append(float(latcomp[idx + 2]))
                                        #print('The energy released per fission is ' + str(assembly['ERC']))
                                        assembly['U235'].append(float(latcomp[idx + 3]))
                                        #print('Assembly U235' + str(assembly['U235']))
                                        assembly['U238'].append(float(latcomp[idx + 4]))
                                        assembly['Pu239'].append(float(latcomp[idx + 5]))
                                        assembly['Pu241'].append(float(latcomp[idx + 6]))
                                        assembly['nubar'].append(float(latcomp[idx + 7]))
                                        assembly['eubound'].append([float(x) for x in latcomp[(idx+8):(idx+(cycle.groups+7))]])
                                        #print('Energy groups upper bounds ' + str(assembly['eubound']))
                                        assembly['chi'].append([float(x) for x in latcomp[(idx+(cycle.groups+7)):(idx+(cycle.groups*2+6))]])
                                        break

                elif (lattice in cycle.refllat):
                    bp_index += 1
                    assembly['nubar'].append(0.0)
                    assembly['ERC'].append(0.0)
                    assembly['bu_clos'].append(0.0)
                    assembly['U235'].append(0)
                    assembly['U238'].append(0)
                    assembly['Pu239'].append(0)
                    assembly['Pu241'].append(0)

            if (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                if assembly['r-f-d'] == 'FUEL':
                    assembly['chi']= np.mean(assembly['chi'], 0) #average to get one value per spectrum    
                    assembly['eubound']= np.mean(assembly['eubound'], 0) #average to get one value per spectrum
                    #print(assembly['eubound'])
                    #print(assembly['chi'])
        
        #PLOTS OUTPUT
        plotpath = Path(os.path.join(out.base_dir, 'plots/ass_source'))
        plotpath.mkdir(parents=True, exist_ok=True)

        #SANITY CHECKS
        checkspath = Path(os.path.join(out.base_dir, 'sanity_checks/ass_source'))
        checkspath.mkdir(parents=True, exist_ok=True)


        #SANITY CHECK: plot burnup profile for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['buprofile'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Burnup (MWd/kgHM)')
                plt.title('Assembly-wise burnup profile')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_burnup_comparison_{self.step}.png')
        plt.close()

        #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
        fig = plt.figure(dpi=300, figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = np.mean(assembly['buprofile'][1:(geom.naxial-1)])
                for i in range(1,(geom.naxial-1)):
                    # the values of the burnup profile are represented by the colormap
                    ax.bar3d(assembly['coordinates'][0], assembly['coordinates'][1], i, 1, 1, 1, shade=True, color=plt.cm.plasma(assembly['buprofile'][i]/(geom.naxial-1)), edgecolor= 'black', linewidth= 0.2)
        #create a legend next to the plot which writes the assembly location and average burnup in descending order
        # Create a legend for the plot
        legend_elements = []
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = np.mean(assembly['buprofile'][1:(geom.naxial-1)])
                legend_elements.append(f'Assembly {assembly["coordinates"]}: {avg_burnup:.2f} MWd/kgHM')

        # Add the legend to the plot
        legend_text = "\n".join(legend_elements)
        plt.figtext(0.08, 0.14, legend_text, horizontalalignment='left', fontsize=6, bbox=dict(facecolor='lightgrey', alpha=0.5))
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=(geom.naxial-1))), ax=ax, label='Burnup (MWd/kgHM)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #change the orientation of the graph
        ax.view_init(elev=30, azim=45)
        #plt.title('3D Assembly-wise burnup profile')
        plt.savefig(plotpath / f'3D_assembly_burnup_comparison_{self.step}.png', bbox_inches='tight')
        plt.close()

        # save the data for these assemblies to a csv file
        with open(checkspath / f'burnup_profile_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['buprofile'][i])
                writer.writerow(row)

        #SANITY CHECK: plot U235 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U235'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U235 Atomic Density')
                plt.title('Assembly-wise U235 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_u235_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot U238 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U238'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U238 Atomic Density')
                plt.title('Assembly-wise U238 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_u238_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot PU239 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu239'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu239 Atomic Density')
                plt.title('Assembly-wise PU239 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_pu239_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot PU241 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu241'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu241 Atomic Density')
                plt.title('Assembly-wise PU241 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_pu241_comparison_{self.step}.png')
        plt.close()

        # save plutonium content to txt
        with open(checkspath / f'plutonium_content_{self.step}.txt', 'a', newline='') as f:
            
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    f.write(f'{self.step},')
                    f.write(f'{assembly["coordinates"][0]},')
                    f.write(f'{assembly["coordinates"][1]},')
                    f.write(f'{assembly["U235"][20]},')
                    f.write(f'{assembly["U238"][20]},')
                    f.write(f'{assembly["Pu239"][20]},')
                    f.write(f'{assembly["Pu241"][20]}\n')
        
        # microscopic fission cross-sections (barns)
        sigma_f_235 = 566.0
        sigma_f_238 = 1.5
        sigma_f_239 = 781.0
        sigma_f_241 = 1060.0 

        # neutron yield per fission
        nu_235 = 2.430
        nu_238 = 2.810
        nu_239 = 2.871
        nu_241 = 2.969

        # energy recoverable from fission
        erf_235 = 201.7
        erf_238 = 205.0
        erf_239 = 210.0
        erf_241 = 212.4

        # Constants
        C = 1.6019e-13 # MeV to Joules

        # Define fission spectrum functions
        def chi_235(E):
            return np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238(E):
            return np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239(E):
            return np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241(E):
            return np.sqrt(E) * np.exp(-E / 1.360)

        # Energy bins - defined center points
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])  # Midpoints of bins

        # integrate raw fission spectrum
        print('computing spectrum integral for normalization ...')
        integral5, error = quad(chi_235, 0, 20)
        print(f"U-235 Integral result: {integral5}, Estimated error: {error}")   
        integral8, error = quad(chi_238, 0, 20)
        print(f"U-238 Integral result: {integral8}, Estimated error: {error}")
        integral9, error = quad(chi_239, 0, 20)
        print(f"PU-239 Integral result: {integral9}, Estimated error: {error}")
        integral41, error = quad(chi_241, 0, 20)
        print(f"PU-241 Integral result: {integral41}, Estimated error: {error}")    

        # Define fission spectrum functions
        def chi_235_norm(E):
            return 1/integral5 * np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238_norm(E):
            return 1/integral8* np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239_norm(E):
            return 1/integral9* np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241_norm(E):
            return 1/integral41* np.sqrt(E) * np.exp(-E / 1.360)

        # integrate raw fission spectrum
        print('checking new integration ...')
        integralcheck5, error = quad(chi_235_norm, 0, 20)
        print(f"U-235 Integral result: {integralcheck5}, Estimated error: {error}")   
        integralcheck8, error = quad(chi_238_norm, 0, 20)
        print(f"U-238 Integral result: {integralcheck8}, Estimated error: {error}")
        integralcheck9, error = quad(chi_239_norm, 0, 20)
        print(f"PU-239 Integral result: {integralcheck9}, Estimated error: {error}")
        integralcheck41, error = quad(chi_241_norm, 0, 20)
        print(f"PU-241 Integral result: {integralcheck41}, Estimated error: {error}") 

        # Calculate histogram values by evaluating the function at bin centers
        hist_235 = chi_235_norm(bin_centers)
        hist_238 = chi_238_norm(bin_centers)
        hist_239 = chi_239_norm(bin_centers)
        hist_241 = chi_241_norm(bin_centers)

        print('Step 8.2: preparing power to source conversion factors ...')

        #prepare plot for spectrum
        plt.figure(dpi=300, figsize=(10, 6))

        #finish building factors
        for assembly in assembly_list:
            if assembly['r-f-d'] == 'FUEL':

                # first node is a reflector
                assembly['nu'].append(0.0)
                assembly['erf'].append(0.0)
                assembly['F'].append(0.0)

                for i in range(1,(geom.naxial-1)):

                    # average assembly neutron multiplication factor (axially discretized)
                    assembly['nu'].append(np.sum(nu_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                    # average assembly energy recoverable from fission (axially discretized)
                    assembly['erf'].append(np.sum(erf_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                    # assembly conversion factor (axially discretized)
                    if (cycle.polarisoption == 0) or (cycle.polarisoption == 2):
                        print("Polaris option selected: using nubar and ERF from literature assumptions")
                        assembly['F'].append(assembly['nu'][i]/(assembly['erf'][i]*C))
                    elif (cycle.polarisoption == 1) or (cycle.polarisoption == 3):
                        print("Polaris option selected: using nubar and ERF from lattice library")
                        assembly['F'].append(assembly['nubar'][i]/(assembly['ERC'][i]*C))

                if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # spectrum from assumptions

                    for j in range(len(bin_centers)):
                        # average assembly fission spectrum (1 spectrum per assembly)
                        assembly['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_238[j]*(nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_239[j]*(nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_241[j]*(nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))))
                
                    print('Normalizing weighted fission spectrum ...')
                    # integrate raw fission spectrum
                    bin_widths = np.diff(self.bins)
                    integral = np.sum(np.array(assembly['chi']) * bin_widths)
                    #print('Raw weighted fission spectrum integral: ' + str(integral))
                    #normalize fission spectrum
                    assembly['chi'] = np.array(assembly['chi'])/integral
                    #check if normalization worked
                    integral= np.sum(np.array(assembly['chi'])*bin_widths)
                    #print('Normalized weighted fission spectrum integral: ' + str(integral))

                # last node is a reflector
                assembly['nu'].append(0.0)
                assembly['erf'].append(0.0)
                assembly['F'].append(0.0)

                # SANITY CHECK: plot fission spectrum
                if (assembly['coordinates'] in geom.source):
                    if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # polaris option 0 and option 1
                        plt.step(bin_centers, assembly['chi'], label= 'Assembly ' + str(assembly['coordinates']))
                    elif (cycle.polarisoption == 2) or (cycle.polarisoption == 3): # polaris option 1 and option 3
                        plt.step(assembly['eubound'], assembly['chi'], label= 'Assembly ' + str(assembly['coordinates']))
                    plt.legend(fontsize=3)
                    plt.xlabel('Energy [MeV]')
                    plt.ylabel('Probability(-)')
                    plt.title('Weighted Fission Spectrum')
        plt.grid()
        plt.savefig(plotpath / f'fission_spectrum_{self.step}.png')
        plt.close()

        # Save fission spectrum to CSV
        if (cycle.polarisoption==0) or (cycle.polarisoption==1) :
            with open(checkspath / f'fission_spectrum_{self.step}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = ['Energy (MeV)'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
                writer.writerow(headers)
                for i in range(len(bin_centers)):
                    row = [bin_centers[i]]
                    for assembly in assembly_list:
                        if assembly['coordinates'] in geom.source:
                            row.append(assembly['chi'][i])
                    writer.writerow(row)
                                        
        print('Computing source ...')
        c= 0

        # iterate on each pin to get source term
        for ass in cycle.asspowerdata:
            if ([ass[1], ass[2]] in geom.source) and (ass[0] == self.step+1): # STEP CONTROL
                c += 1
                print('Computing source for source assembly n. ' + str(c))
                for assembly in assembly_list:
                    if (ass[1], ass[2]) == (assembly['coordinates'][0], assembly['coordinates'][1]):
                        # flip assembly to get factor from bottom to top
                        factor= np.flip(assembly['F']) 
                        # extract fuel type for source definition
                        fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]
                                  
                        if self.truncoption == False:
                            # multiply to get source term and overwrite  pin data
                            # case number (burnup id), index i (assembly x), index j (assembly y), index k (assembly z), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum), fuel type
                            self.sourceassy.append((ass[0], ass[1], ass[2], ass[3], ass[4], ass[4]*factor[ass[3]-1], assembly['chi'], fueltype, assembly['eubound']))
                            break

                        elif self.truncoption == True:
                            if ([ass[1], ass[2]] in self.trunc_ass_X) or (([ass[1], ass[2]] in self.trunc_ass_Y)):
                                self.sourceassy.append((ass[0], ass[1], ass[2], ass[3], ass[4], ass[4]*factor[ass[3]-1]/2, assembly['chi'], fueltype, assembly['eubound']))
                                break

                            else:
                                self.sourceassy.append((ass[0], ass[1], ass[2], ass[3], ass[4], ass[4]*factor[ass[3]-1], assembly['chi'], fueltype, assembly['eubound']))
                                break

    def write_source_ass(
             self, cycle: "Cycle", geom: "Geometry", out: "Outputs", filepath: Union[str, Path]) -> None:
        """
        Write assembly-level neutron source terms to external source file format.
        
        This method outputs the computed assembly source data to Serpent-compatible external source
        specification files. It handles spatial binning (volume vs point sources), energy spectrum
        output, and applies truncation geometry transformations if specified.
        
        Parameters
        ----------
        cycle : Cycle
            Cycle object containing polaris options and energy group information.
        geom : Geometry
            Geometry object containing core mesh, assembly pitch, and z-coordinate information.
        out : Outputs
            Outputs object for managing output directories.
        filepath : Union[str, Path]
            Path to the Serpent main input file to be updated with source strength normalization.
        
        Returns
        -------
        None
            Writes external source files to the neutron_source directory and updates the main
            Serpent input file with the total source strength.
        """

        # define output directory for source
        outpath = Path(os.path.join(out.base_dir, 'neutron_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        print('Writing assembly source term ...')
        source_sum= 0
        index= 0

        z_core_flip= np.flip(geom.z_core)

        # compute total source strength
        for ass in self.sourceassy:

            # sum source of neutrons
            source_sum += ass[5]

        with open(os.path.join(outpath,'LWR-10-external_source_' + str(self.step) + 'AWS.ser'), 'w') as f:
            

            for ass in self.sourceassy:
                index += 1

                # write source for pin
                print('Writing source for assembly ...' + str(index))

                # compute assembly center coordinates
                x_core = (ass[2] - (geom.nass + 1) / 2) * geom.ass_pitch
                y_core = ((geom.nass + 1) / 2 - ass[1]) * geom.ass_pitch
                z_core = z_core_flip[ass[3]-1]
                node_height= geom.meshheight[ass[3]-1]
            
                # compute assembly boundaries
                x_min= x_core - geom.ass_pitch/2
                x_max= x_core + geom.ass_pitch/2
                y_min= y_core - geom.ass_pitch/2
                y_max= y_core + geom.ass_pitch/2
                z_min= z_core - node_height/2
                z_max= z_core + node_height/2

                # fuel type
                fuel_type= ass[7]

                # write neutron source 

                if (self.truncoption == False) :
                    if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # write spectrum from assumptions, reading from len(self.bins) 
                        f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                    if (cycle.polarisoption == 2) or (cycle.polarisoption == 3): # write spectrum from Polaris, reading from cycle.groups
                        f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {cycle.groups} 1\n")
                
                elif (self.truncoption == True) :

                    if ([ass[1], ass[2]] in self.trunc_ass_X):
                        if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # write spectrum from assumptions, reading from len(self.bins) 
                            f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_core:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                        if (cycle.polarisoption == 2) or (cycle.polarisoption == 3): # write spectrum from Polaris, reading from cycle.groups
                            f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_core:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {cycle.groups} 1\n")
                    
                    elif ([ass[1], ass[2]] in self.trunc_ass_Y):
                        if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # write spectrum from assumptions, reading from len(self.bins) 
                            f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_core:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                        if (cycle.polarisoption == 2) or (cycle.polarisoption == 3): # write spectrum from Polaris, reading from cycle.groups
                            f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_core:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {cycle.groups} 1\n")

                    else :
                        if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # write spectrum from assumptions, reading from len(self.bins) 
                            f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                        if (cycle.polarisoption == 2) or (cycle.polarisoption == 3): # write spectrum from Polaris, reading from cycle.groups
                            f.write(f"\nsrc {int(index)} n sw {ass[5]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {cycle.groups} 1\n")
                      
                f.write('1E-11 0.0\n') # insert first bin with zero value
                #insert energy spectrum specification
                print('Writing energy spectrum for assembly ...' + str(index))
                if (cycle.polarisoption == 0) or (cycle.polarisoption == 1): # write spectrum from assumptions
                    for i in range(len(self.bins) - 1):
                        f.write(f"{self.bins[i+1]:.5e} {ass[6][i]:.5e}\n")
                elif (cycle.polarisoption == 2) or (cycle.polarisoption == 3): # write spectum from polaris
                    for i in range(cycle.groups - 1):
                        f.write(f"{ass[8][i]:.5e} {ass[6][i]:.5e}\n")

        with open(os.path.join(filepath), 'r') as f:
            lines = f.readlines()

        print ('Writing total source strength for normalization ...')
        with open(os.path.join(outpath, 'LWR-09-main_' + str(self.step) + 'AWS.ser'), 'w') as f:
            for i, line in enumerate(lines):
                if 'set srcrate' in line:
                    lines[i] = f'set srcrate {source_sum:.5e}\n'
                    break
            f.writelines(lines)

    def build_source_pin(
                    self, cycle: "Cycle", geom: "Geometry", out: "Outputs") -> None:
        """
        Build pin-level neutron source terms from cycle and geometry data.
        
        This method processes pin-wise burnup profiles, compositions, and generates
        neutron multiplication factors and fission spectra at the pin level. It computes
        conversion factors from power to source terms and applies truncation options if
        specified. Pin-level source modeling provides higher spatial resolution compared
        to assembly-level approximations.
        
        Parameters
        ----------
        cycle : Cycle
            Cycle object containing pin power data, exposure profiles, lattice compositions,
            and interpolation options.
        geom : Geometry
            Geometry object containing assembly and core mesh information, pin pitch,
            and axial discretization details.
        out : Outputs
            Outputs object for managing output directories and file paths for plots
            and sanity check outputs.
        
        Returns
        -------
        None
            Populates self.sourcepin with computed pin-level source data. Each entry
            contains: (step, i_coord, j_coord, k_coord, pin_x, pin_y, power, source_term,
            spectrum, fuel_type).
        
        Notes
        -----
        - Uses microscopic fission cross-sections and neutron yields from literature.
        - Generates normalized fission spectra using Watt and Maxwell distribution functions.
        - Supports both interpolated and non-interpolated pin power data via cycle.interpoption.
        - Handles truncation geometry with symmetry axis considerations for pin-level divisions.
        - Polaris options 2 and 3 (with energy group-dependent data) are not supported
          for pin-wise source modeling.
        
        Raises
        ------
        SystemExit
            If Polaris options 2 or 3 are selected, as these are incompatible with
            pin-wise source calculations.
        
        See Also
        --------
        build_source_ass : Assembly-level source term generation.
        write_source_pin : Output pin-level source terms to external file format.
        """

        print('Step 8.1: preparing data for source term ...')

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(geom.naxial,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[]}
            assembly_list.append(assembly)

        # define which assemblies are fuel and which are reflector or dummies
        for assembly in assembly_list:
            if assembly['assytype'] == 0:
                assembly['r-f-d']= 'DUMMY'
            elif assembly['assytype'] != 0:
                for axial in cycle.assyaxial:
                    if axial[1] == assembly['assytype']:
                        assembly['r-f-d']= axial[0]

        # extract the burnup ID from exposure info
        for assembly in assembly_list:
            for assy in cycle.assyexp:
                if assembly['coordinates'] == [assy[0], assy[1]]:
                    assembly['burnupID']= assy[2]

        # extract the burnup profile from last exposure point for each assembly
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                assembly['buprofile'] = cycle.exposure[:, assembly['burnupID'] - 1, self.step].flatten().tolist() #USER INPUT: currenly extracting specific burnup step: assemblies which have burnupID = 0 will have a burnup profile of 0

        # extract the lattice configuration of each assembly type
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                for axialinfo in cycle.assyaxial:
                    if axialinfo[1] == assembly['assytype']:
                        assembly['latID']= axialinfo[2]
            elif assembly['burnupID'] == []: #preassign values to reflector assembly types
                assembly['latID']= [0]*geom.naxial
                assembly['burnupID']= 0
                assembly['buprofile']= [0]*geom.naxial
                assembly['bu_clos']= [0]*geom.naxial
                assembly['nubar']= 0
                assembly['ERC']= 0
                assembly['U235']= [0]*geom.naxial
                assembly['U238']= [0]*geom.naxial
                assembly['Pu239']= [0]*geom.naxial
                assembly['Pu241']= [0]*geom.naxial
        
        # extract the specific lattice composition: find the corresponding burnup point in lattice library and the composition
        for assembly in assembly_list:
            bp_index= -1 
            for lattice in assembly['latID']:
                if (lattice in cycle.fuellat):
                    bp_index += 1
                    for latcomp in cycle.latcomp:
                        if (lattice == int(float(latcomp[0]))): # only fuel lattices should look for uranium and plutonium composition
                            #print('Burnup index is' + str(bp_index))
                            bppoint = assembly['buprofile'][bp_index]
                            closest_burnup = min(cycle.latburn, key=lambda x: abs(x - bppoint))
                            assembly['bu_clos'].append(closest_burnup)

                            if (cycle.polarisoption == 0) or (cycle.polarisoption == 1):
                                for idx in range(0, len(latcomp), 7):
                                    if float(latcomp[idx + 1]) == closest_burnup:
                                        assembly['ERC'].append(float(latcomp[idx + 2]))
                                        assembly['U235'].append(float(latcomp[idx + 3]))
                                        assembly['U238'].append(float(latcomp[idx + 4]))
                                        assembly['Pu239'].append(float(latcomp[idx + 5]))
                                        assembly['Pu241'].append(float(latcomp[idx + 6]))
                                        assembly['nubar'].append(float(latcomp[idx + 7]))
                                        break

                            if (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                                print('Polaris Option 2 or 3 are currently not available for pin-wise source modeling.')
                                raise SystemExit

                elif (lattice in cycle.refllat):
                    bp_index += 1
                    assembly['nubar'].append(0.0)
                    assembly['ERC'].append(0.0)
                    assembly['bu_clos'].append(0.0)
                    assembly['U235'].append(0)
                    assembly['U238'].append(0)
                    assembly['Pu239'].append(0)
                    assembly['Pu241'].append(0)

        #PLOTS OUTPUT
        plotpath = Path(os.path.join(out.base_dir, 'plots/pin_source'))
        plotpath.mkdir(parents=True, exist_ok=True)

        #SANITY CHECKS
        checkspath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        checkspath.mkdir(parents=True, exist_ok=True)

        #SANITY CHECK: plot burnup profile for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['buprofile'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Burnup (MWd/kgHM)')
                plt.title('Assembly-wise burnup profile')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_burnup_comparison_{self.step}.png')
        plt.close()

        #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
        fig = plt.figure(dpi=300, figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = np.mean(assembly['buprofile'][1:(geom.naxial-1)])
                for i in range(1, geom.naxial-1):
                    # the values of the burnup profile are represented by the colormap
                    ax.bar3d(assembly['coordinates'][0], assembly['coordinates'][1], i, 1, 1, 1, shade=True, color=plt.cm.plasma(assembly['buprofile'][i]/(geom.naxial-1)), edgecolor= 'black', linewidth= 0.2)
        #create a legend next to the plot which writes the assembly location and average burnup in descending order
        # Create a legend for the plot
        legend_elements = []
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = np.mean(assembly['buprofile'][1:(geom.naxial-1)])
                legend_elements.append(f'Assembly {assembly["coordinates"]}: {avg_burnup:.2f} MWd/kgHM')

        # Add the legend to the plot
        legend_text = "\n".join(legend_elements)
        plt.figtext(0.08, 0.14, legend_text, horizontalalignment='left', fontsize=6, bbox=dict(facecolor='lightgrey', alpha=0.5))
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=60)), ax=ax, label='Burnup (MWd/kgHM)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #change the orientation of the graph
        ax.view_init(elev=30, azim=45)
        #plt.title('3D Assembly-wise burnup profile')
        plt.savefig(plotpath / f'3D_assembly_burnup_comparison_{self.step}.png', bbox_inches='tight')
        plt.close()

        # save the data for these assemblies to a csv file
        with open(checkspath / f'burnup_profile_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['buprofile'][i])
                writer.writerow(row)

        #SANITY CHECK: save U235 composition for each assembly
        with open(checkspath + f'u235_composition_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in asssource]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [geom.naxial - i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['U235'][i])
                writer.writerow(row)

        #SANITY CHECK: save U238 composition for each assembly
        with open(checkspath + f'u238_composition_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in asssource]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [geom.naxial - i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['U238'][i])
                writer.writerow(row)

        #SANITY CHECK: save PU239 composition for each assembly
        with open(checkspath + f'pu239_composition_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in asssource]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [geom.naxial - i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['Pu239'][i])
                writer.writerow(row)
        
        #SANITY CHECK: save PU241 composition for each assembly
        with open(checkspath + f'pu241_composition_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in asssource]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [geom.naxial - i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['Pu241'][i])
                writer.writerow(row)

        #SANITY CHECK: plot U235 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U235'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U235 Atomic Density')
                plt.title('Assembly-wise U235 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_u235_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot U238 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U238'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U238 Atomic Density')
                plt.title('Assembly-wise U238 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_u238_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot PU239 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu239'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu239 Atomic Density')
                plt.title('Assembly-wise PU239 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_pu239_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot PU241 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu241'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu241 Atomic Density')
                plt.title('Assembly-wise PU241 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_pu241_comparison_{self.step}.png')
        plt.close()

        # extract the factors necessary for the source term

        # microscopic fission cross-sections (barns)
        sigma_f_235 = 566.0
        sigma_f_238 = 1.5
        sigma_f_239 = 781.0
        sigma_f_241 = 1060.0 

        # neutron yield per fission
        nu_235 = 2.430 # MCNP-DATA  # ENDFBVIII.0 = 2.414000 
        nu_238 = 2.810 # MCNP-DATA  # ENDFBVIII.0 = 2.611820
        nu_239 = 2.871 # MCNP-DATA  # ENDFBVIII.0 = 2.868503 
        nu_241 = 2.969 # MCNP-DATA  # ENDFBVIII.0 = 2.929100

        # energy recoverable from fission
        erf_235 = 201.7
        erf_238 = 205.0
        erf_239 = 210.0
        erf_241 = 212.4

        # Constants
        E = np.linspace(0, 20, 1000)
        C = 1.6019e-13 # MeV to Joules

        # Define fission spectrum functions (Watt and Maxwell)
        def chi_235(E):
            return np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238(E):
            return np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239(E):
            return np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241(E):
            return np.sqrt(E) * np.exp(-E / 1.360)

        # Energy bins - defined center points
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])  # Midpoints of bins

        # integrate raw fission spectrum
        print('computing spectrum integral for normalization ...')
        integral5, error = quad(chi_235, 0, 20)
        print(f"U-235 Integral result: {integral5}, Estimated error: {error}")   
        integral8, error = quad(chi_238, 0, 20)
        print(f"U-238 Integral result: {integral8}, Estimated error: {error}")
        integral9, error = quad(chi_239, 0, 20)
        print(f"PU-239 Integral result: {integral9}, Estimated error: {error}")
        integral41, error = quad(chi_241, 0, 20)
        print(f"PU-241 Integral result: {integral41}, Estimated error: {error}")    

        # Define fission spectrum functions
        def chi_235_norm(E):
            return 1/integral5 * np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238_norm(E):
            return 1/integral8* np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239_norm(E):
            return 1/integral9* np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241_norm(E):
            return 1/integral41* np.sqrt(E) * np.exp(-E / 1.360)

        # integrate raw fission spectrum
        print('checking new integration ...')
        integralcheck5, error = quad(chi_235_norm, 0, 20)
        print(f"U-235 Integral result: {integralcheck5}, Estimated error: {error}")   
        integralcheck8, error = quad(chi_238_norm, 0, 20)
        print(f"U-238 Integral result: {integralcheck8}, Estimated error: {error}")
        integralcheck9, error = quad(chi_239_norm, 0, 20)
        print(f"PU-239 Integral result: {integralcheck9}, Estimated error: {error}")
        integralcheck41, error = quad(chi_241_norm, 0, 20)
        print(f"PU-241 Integral result: {integralcheck41}, Estimated error: {error}") 

        # Calculate histogram values by evaluating the function at bin centers
        hist_235 = chi_235_norm(bin_centers)
        hist_238 = chi_238_norm(bin_centers)
        hist_239 = chi_239_norm(bin_centers)
        hist_241 = chi_241_norm(bin_centers)

        print('Step 8.2: preparing power to source conversion factors ...')

        #prepare plot for spectrum
        plt.figure(dpi=300, figsize=(10, 6))

        #finish building factors
        for assembly in assembly_list:
            if assembly['r-f-d'] == 'FUEL':

                # first node is a reflector
                assembly['nu'].append(0.0)
                assembly['erf'].append(0.0)
                assembly['F'].append(0.0)

                for i in range(1, geom.naxial-1):

                    # average assembly neutron multiplication factor (axially discretized)
                    assembly['nu'].append(np.sum(nu_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                    # average assembly energy recoverable from fission (axially discretized)
                    assembly['erf'].append(np.sum(erf_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                    # assembly conversion factor (axially discretized)
                    if (cycle.polarisoption == 0):
                        print("Polaris option selected: using nubar and ERF from literature assumptions")
                        assembly['F'].append(assembly['nu'][i]/(assembly['erf'][i]*C))
                    elif (cycle.polarisoption == 1):
                        print("Polaris option selected: using nubar and ERF from lattice library")
                        assembly['F'].append(assembly['nubar'][i]/(assembly['ERC'][i]*C))



                for j in range(len(bin_centers)):
                    # average assembly fission spectrum (1 spectrum per assembly)
                        assembly['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_238[j]*(nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_239[j]*(nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_241[j]*(nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))))
                
                print('Normalizing weighted fission spectrum ...')
                # integrate raw fission spectrum
                bin_widths = np.diff(self.bins)
                integral = np.sum(np.array(assembly['chi']) * bin_widths)
                #print('Raw weighted fission spectrum integral: ' + str(integral))
                #normalize fission spectrum
                assembly['chi'] = np.array(assembly['chi'])/integral
                #check if normalization worked
                integral= np.sum(np.array(assembly['chi'])*bin_widths)
                #print('Normalized weighted fission spectrum integral: ' + str(integral))

                # last node is a reflector
                assembly['nu'].append(0.0)
                assembly['erf'].append(0.0)
                assembly['F'].append(0.0)

                # SANITY CHECK: plot fission spectrum
                if (assembly['coordinates'] in geom.source):
                    plt.step(bin_centers, assembly['chi'], label= 'Assembly ' + str(assembly['coordinates']))
                    plt.legend(fontsize=3)
                    #plt.xscale('log')
                    plt.xlabel('Energy [MeV]')
                    plt.ylabel('Probability(-)')
                    plt.title('Weighted Fission Spectrum')
        plt.grid()
        plt.savefig(os.path.join(plotpath, f'fission_spectrum_{self.step}.png'))
        plt.close()

        # Save fission spectrum to CSV
        with open(os.path.join(checkspath, f'fission_spectrum_{self.step}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Energy (MeV)'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
            writer.writerow(headers)
            for i in range(len(bin_centers)):
                row = [bin_centers[i]]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['chi'][i])
                writer.writerow(row)
                                            
        print('Step 8.3: computing source ...')
        c= 0


        if cycle.interpoption:
            print('Interpolation Option Activated ...')
            # iterate on each pin to get source term
            for pin in cycle.pinpowerdatainterp:
                if ([pin[1], pin[2]] in geom.source) and (pin [0] == self.step+1):
                    c += 1
                    print('Computing source for assembly layer n. ' + str(c))
                    for assembly in assembly_list:
                        if (pin[1], pin[2]) == (assembly['coordinates'][0], assembly['coordinates'][1]):

                            if (self.truncoption == False):

                                # flip assembly to get factor from bottom to top
                                factor= np.flip(assembly['F'])
                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                                # multiply to get source term and overwrite  pin data
                                # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                break

                            elif (self.truncoption == True):

                                # flip assembly to get factor from bottom to top
                                factor= np.flip(assembly['F'])     

                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]
                  
                                if ([pin[1], pin[2]] in self.trunc_ass_X):

                                    if ([pin[4],pin[5]] not in self.trunc_pin_X) and ([pin[4],pin[5]] not in self.trunc_pinsym_X):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                        break

                                    if ([pin[4],pin[5]] in self.trunc_pinsym_X): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                        break


                                elif ([pin[1], pin[2]] in self.trunc_ass_Y):
                                    if ([pin[4],pin[5]] not in self.trunc_pin_Y) and ([pin[4],pin[5]] not in self.trunc_pinsym_Y):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                        break
                                    
                                    if ([pin[4],pin[5]] in self.trunc_pinsym_Y): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                        break

                                else:
                                    self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                    break

        else:
            # iterate on each pin to get source term
            for pin in cycle.pinpowerdata:
                if ([pin[1], pin[2]] in geom.source) and (pin [0] == self.step+1) and (pin[6] != 0): # only consider locations with non-zero power:
                    c += 1
                    print('Computing source for assembly layer n. ' + str(c))
                    for assembly in assembly_list:
                        if (pin[1], pin[2]) == (assembly['coordinates'][0], assembly['coordinates'][1]):

                            if (self.truncoption == False):

                                # flip assembly to get factor from bottom to top
                                factor= np.flip(assembly['F'])
                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                                # multiply to get source term and overwrite  pin data
                                # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                break

                            elif (self.truncoption == True):

                                # flip assembly to get factor from bottom to top
                                factor= np.flip(assembly['F'])     

                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]
                       

                                if ([pin[1], pin[2]] in self.trunc_ass_X):

                                    if ([pin[4],pin[5]] not in self.trunc_pin_X) and ([pin[4],pin[5]] not in self.trunc_pinsym_X):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                        break

                                    if ([pin[4],pin[5]] in self.trunc_pinsym_X): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                        break


                                elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                                    if ([pin[4],pin[5]] not in self.trunc_pin_Y) and ([pin[4],pin[5]] not in self.trunc_pinsym_Y):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                        break
                                    
                                    if ([pin[4],pin[5]] in self.trunc_pinsym_Y): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                        break

                                else:
                                    self.sourcepin.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                    break


        # save total source rate to check file
        with open(os.path.join(checkspath, 'source_rate_pin3D.txt'), 'w', newline='') as f:
            f.write('Total source rate: ' + str(np.sum([x[7] for x in self.sourcepin])))

    def write_source_pin(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs", filepath: Union[str, Path]) -> None:
        """
        Write pin-level neutron source terms to external source file format.
        
        This method outputs the computed pin-level source data to Serpent-compatible external source
        specification files. It handles spatial binning for both point and volume sources, applies
        energy spectrum output, and implements truncation geometry transformations if specified.
        Pin-level sources provide higher spatial resolution than assembly-level approximations.
        
        Parameters
        ----------
        cycle : Cycle
            Cycle object containing polaris options and energy group information.
        geom : Geometry
            Geometry object containing core mesh, assembly pitch, pin pitch, z-coordinates,
            and mesh height information for spatial coordinate calculations.
        out : Outputs
            Outputs object for managing output directories and file paths.
        filepath : Union[str, Path]
            Path to the Serpent main input file to be updated with source strength normalization.
        
        Returns
        -------
        None
            Writes external source files to the neutron_source directory and updates the main
            Serpent input file with the total source strength.
        
        Notes
        -----
        - Creates two output files per step: external source specification (PWS.ser) and 
          updated main input file (main_*PWS.ser).
        - Supports both 'pointsource' and 'volumesource' options via self.sourceoption.
        - Point sources are defined as single coordinates (sp syntax).
        - Volume sources are defined as spatial bins (sx, sy, sz syntax) with material specification.
        - For truncation geometry, applies spatial restrictions to sources on symmetry axes.
        - Energy spectrum is output using self.bins for bin edges and computed spectrum values.
        - Normalizes source weights by total source strength for probabilistic sampling.
        - Pins on truncation symmetry axes (trunc_pinsym_X, trunc_pinsym_Y) have restricted
          volume bounds to one side of the symmetry plane.
        
        Raises
        ------
        FileNotFoundError
            If the input filepath for the main Serpent file does not exist.
        IOError
            If output directory cannot be created or files cannot be written.
        
        See Also
        --------
        build_source_pin : Pin-level source term generation.
        write_source_ass : Assembly-level source term output.
        
        Examples
        --------
        >>> source = Source(step=0, bins=np.logspace(-2, 1, 50), truncoption=False,
        ...                 assytype_to_mcmaterial={'1': 'fuel1'}, sourceoption='pointsource')
        >>> source.build_source_pin(cycle, geom, out)
        >>> source.write_source_pin(cycle, geom, out, 'path/to/main.ser')
        """


        # define output directory for source
        outpath = Path(os.path.join(out.base_dir, 'neutron_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        print('Writing source term ...')
        source_sum= 0
        index= 0

        z_core_flip= np.flip(geom.z_core)

        # compute total source strength
        for pin in self.sourcepin:

            # sum source of neutrons
            source_sum += pin[7]

        with open(os.path.join(outpath,'LWR-10-external_source_' + str(self.step) + 'PWS.ser'), 'w') as f:
            
            for pin in self.sourcepin:

                index += 1

                # write source for pin
                print('Writing source for pin ...' + str(index))

                # compute pin coordinates (UPDATED)
                cA = (geom.nass + 1) / 2    # per 15 -> 8.0
                cP = (geom.npin + 1) / 2    # per 14 -> 7.5
                x_core = (pin[2] - cA) * geom.ass_pitch
                y_core = (cA - pin[1]) * geom.ass_pitch
                x_pin = x_core + (pin[5] - cP) * geom.pin_pitch
                y_pin = y_core + (cP - pin[4]) * geom.pin_pitch
                z_pin = z_core_flip[pin[3]-1]
                node_height= geom.meshheight[pin[3]-1]
                
                # compute pin boundaries
                x_min= x_pin - geom.pin_pitch/2
                x_max= x_pin + geom.pin_pitch/2
                y_min= y_pin - geom.pin_pitch/2
                y_max= y_pin + geom.pin_pitch/2
                z_min= z_pin - node_height/2
                z_max= z_pin + node_height/2

                # fuel type
                fuel_type= pin[9]
                # write neutron source 
                if self.sourceoption=='pointsource':
                
                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X):
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                if self.sourceoption=='volumesource':

                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                    
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X):
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_pin:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                        elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_pin:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                           
                f.write('1E-11 0.0\n') # insert first bin with zero value
                #insert energy spectrum specification
                print('Writing energy spectrum for pin ...' + str(index))
                for i in range(len(self.bins) - 1):
                    f.write(f"{self.bins[i+1]:.5e} {pin[8][i]:.5e}\n")
            
        with open(os.path.join(filepath), 'r') as f:
            lines = f.readlines()

        print ('Writing total source strength for normalization ...')
        with open(os.path.join(outpath, 'LWR-09-main_' + str(self.step) + 'PWS.ser'), 'w') as f:
            for i, line in enumerate(lines):
                if 'set srcrate' in line:
                    lines[i] = f'set srcrate {source_sum:.5e}\n'
                    break
            f.writelines(lines)

    def build_source_pin2D(
                    self, cycle: "Cycle", geom: "Geometry", out: "Outputs") -> None:
        """
        Build 2D pin-level neutron source terms from cycle and geometry data.
        
        This method processes pin-wise burnup profiles, compositions, and generates
        neutron multiplication factors and fission spectra at the pin level. It computes
        conversion factors from power to source terms and applies truncation options if
        specified. Pin-level source modeling provides higher spatial resolution compared
        to assembly-level approximations.
        
        Parameters
        ----------
        cycle : Cycle
            Cycle object containing pin power data, exposure profiles, lattice compositions,
            and interpolation options.
        geom : Geometry
            Geometry object containing assembly and core mesh information, pin pitch,
            and axial discretization details.
        out : Outputs
            Outputs object for managing output directories and file paths for plots
            and sanity check outputs.
        
        Returns
        -------
        None
            Populates self.sourcepin with computed pin-level source data. Each entry
            contains: (step, i_coord, j_coord, k_coord, pin_x, pin_y, power, source_term,
            spectrum, fuel_type).
        
        Notes
        -----
        - Uses microscopic fission cross-sections and neutron yields from literature.
        - Generates normalized fission spectra using Watt and Maxwell distribution functions.
        - Supports both interpolated and non-interpolated pin power data via cycle.interpoption.
        - Handles truncation geometry with symmetry axis considerations for pin-level divisions.
        - Polaris options 2 and 3 (with energy group-dependent data) are not supported
          for pin-wise source modeling.
        
        Raises
        ------
        SystemExit
            If Polaris options 2 or 3 are selected, as these are incompatible with
            pin-wise source calculations.
        
        See Also
        --------
        build_source_ass : Assembly-level source term generation.
        write_source_pin : Output pin-level source terms to external file format.
        """

        print('Step 8.1: preparing data for source term ...')

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(geom.naxial,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[]}
            assembly_list.append(assembly)

        # define which assemblies are fuel and which are reflector or dummies
        for assembly in assembly_list:
            if assembly['assytype'] == 0:
                assembly['r-f-d']= 'DUMMY'
            elif assembly['assytype'] != 0:
                for axial in cycle.assyaxial:
                    if axial[1] == assembly['assytype']:
                        assembly['r-f-d']= axial[0]

        # extract the burnup ID from exposure info
        for assembly in assembly_list:
            for assy in cycle.assyexp:
                if assembly['coordinates'] == [assy[0], assy[1]]:
                    assembly['burnupID']= assy[2]

        # extract the burnup profile from last exposure point for each assembly
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                assembly['buprofile'] = cycle.exposure[:, assembly['burnupID'] - 1, self.step].flatten().tolist() #USER INPUT: currenly extracting specific burnup step: assemblies which have burnupID = 0 will have a burnup profile of 0

        # extract the lattice configuration of each assembly type
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                for axialinfo in cycle.assyaxial:
                    if axialinfo[1] == assembly['assytype']:
                        assembly['latID']= axialinfo[2]
            elif assembly['burnupID'] == []: #preassign values to reflector assembly types
                assembly['latID']= [0]*geom.naxial
                assembly['burnupID']= 0
                assembly['buprofile']= [0]*geom.naxial
                assembly['bu_clos']= [0]*geom.naxial
                assembly['nubar']= 0
                assembly['ERC']= 0
                assembly['U235']= [0]*geom.naxial
                assembly['U238']= [0]*geom.naxial
                assembly['Pu239']= [0]*geom.naxial
                assembly['Pu241']= [0]*geom.naxial
        
        # extract the specific lattice composition: find the corresponding burnup point in lattice library and the composition
        for assembly in assembly_list:
            bp_index= -1 
            for lattice in assembly['latID']:
                if (lattice in cycle.fuellat):
                    bp_index += 1
                    for latcomp in cycle.latcomp:
                        if (lattice == int(float(latcomp[0]))): # only fuel lattices should look for uranium and plutonium composition
                            #print('Burnup index is' + str(bp_index))
                            bppoint = assembly['buprofile'][bp_index]
                            closest_burnup = min(cycle.latburn, key=lambda x: abs(x - bppoint))
                            assembly['bu_clos'].append(closest_burnup)

                            if (cycle.polarisoption == 0) or (cycle.polarisoption == 1):
                                for idx in range(0, len(latcomp), 7):
                                    if float(latcomp[idx + 1]) == closest_burnup:
                                        assembly['ERC'].append(float(latcomp[idx + 2]))
                                        assembly['U235'].append(float(latcomp[idx + 3]))
                                        assembly['U238'].append(float(latcomp[idx + 4]))
                                        assembly['Pu239'].append(float(latcomp[idx + 5]))
                                        assembly['Pu241'].append(float(latcomp[idx + 6]))
                                        assembly['nubar'].append(float(latcomp[idx + 7]))
                                        break

                            if (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                                print('Polaris Option 2 or 3 are currently not available for pin-wise source modeling.')
                                raise SystemExit

                elif (lattice in cycle.refllat):
                    bp_index += 1
                    assembly['nubar'].append(0.0)
                    assembly['ERC'].append(0.0)
                    assembly['bu_clos'].append(0.0)
                    assembly['U235'].append(0)
                    assembly['U238'].append(0)
                    assembly['Pu239'].append(0)
                    assembly['Pu241'].append(0)

        #PLOTS OUTPUT
        plotpath = Path(os.path.join(out.base_dir, 'plots/pin_source'))
        plotpath.mkdir(parents=True, exist_ok=True)

        #SANITY CHECKS
        checkspath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        checkspath.mkdir(parents=True, exist_ok=True)

        #SANITY CHECK: plot burnup profile for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['buprofile'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Burnup (MWd/kgHM)')
                plt.title('Assembly-wise burnup profile')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_burnup_comparison_{self.step}.png')
        plt.close()

        #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
        fig = plt.figure(dpi=300, figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = np.mean(assembly['buprofile'][0]) # NB: just one axial node in 2D models
                for i in range(geom.naxial): # NB: just one axial node in 2D models
                    # the values of the burnup profile are represented by the colormap
                    ax.bar3d(assembly['coordinates'][0], assembly['coordinates'][1], i, 1, 1, 1, shade=True, color=plt.cm.plasma(assembly['buprofile'][i]), edgecolor= 'black', linewidth= 0.2)
        #create a legend next to the plot which writes the assembly location and average burnup in descending order
        # Create a legend for the plot
        legend_elements = []
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = np.mean(assembly['buprofile'][0]) # NB: just one axial node in 2D models
                legend_elements.append(f'Assembly {assembly["coordinates"]}: {avg_burnup:.2f} MWd/kgHM')

        # Add the legend to the plot
        legend_text = "\n".join(legend_elements)
        plt.figtext(0.08, 0.14, legend_text, horizontalalignment='left', fontsize=6, bbox=dict(facecolor='lightgrey', alpha=0.5))
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=60)), ax=ax, label='Burnup (MWd/kgHM)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #change the orientation of the graph
        ax.view_init(elev=30, azim=45)
        #plt.title('3D Assembly-wise burnup profile')
        plt.savefig(plotpath / f'3D_assembly_burnup_comparison_{self.step}.png', bbox_inches='tight')
        plt.close()

        # save the data for these assemblies to a csv file
        with open(checkspath / f'burnup_profile_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['buprofile'][i])
                writer.writerow(row)

        #SANITY CHECK: plot U235 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U235'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U235 Atomic Density')
                plt.title('Assembly-wise U235 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_u235_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot U238 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U238'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U238 Atomic Density')
                plt.title('Assembly-wise U238 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_u238_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot PU239 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu239'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu239 Atomic Density')
                plt.title('Assembly-wise PU239 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_pu239_comparison_{self.step}.png')
        plt.close()

        #SANITY CHECK: plot PU241 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu241'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu241 Atomic Density')
                plt.title('Assembly-wise PU241 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / f'assembly_pu241_comparison_{self.step}.png')
        plt.close()

        # extract the factors necessary for the source term

        # microscopic fission cross-sections (barns)
        sigma_f_235 = 566.0
        sigma_f_238 = 1.5
        sigma_f_239 = 781.0
        sigma_f_241 = 1060.0 

        # neutron yield per fission
        nu_235 = 2.430 # MCNP-DATA  # ENDFBVIII.0 = 2.414000 
        nu_238 = 2.810 # MCNP-DATA  # ENDFBVIII.0 = 2.611820
        nu_239 = 2.871 # MCNP-DATA  # ENDFBVIII.0 = 2.868503 
        nu_241 = 2.969 # MCNP-DATA  # ENDFBVIII.0 = 2.929100

        # energy recoverable from fission
        erf_235 = 201.7
        erf_238 = 205.0
        erf_239 = 210.0
        erf_241 = 212.4

        # Constants
        E = np.linspace(0, 20, 1000)
        C = 1.6019e-13 # MeV to Joules

        # Define fission spectrum functions (Watt and Maxwell)
        def chi_235(E):
            return np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238(E):
            return np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239(E):
            return np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241(E):
            return np.sqrt(E) * np.exp(-E / 1.360)

        # Energy bins - defined center points
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])  # Midpoints of bins

        # integrate raw fission spectrum
        print('computing spectrum integral for normalization ...')
        integral5, error = quad(chi_235, 0, 20)
        print(f"U-235 Integral result: {integral5}, Estimated error: {error}")   
        integral8, error = quad(chi_238, 0, 20)
        print(f"U-238 Integral result: {integral8}, Estimated error: {error}")
        integral9, error = quad(chi_239, 0, 20)
        print(f"PU-239 Integral result: {integral9}, Estimated error: {error}")
        integral41, error = quad(chi_241, 0, 20)
        print(f"PU-241 Integral result: {integral41}, Estimated error: {error}")    

        # Define fission spectrum functions
        def chi_235_norm(E):
            return 1/integral5 * np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238_norm(E):
            return 1/integral8* np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239_norm(E):
            return 1/integral9* np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241_norm(E):
            return 1/integral41* np.sqrt(E) * np.exp(-E / 1.360)

        # integrate raw fission spectrum
        print('checking new integration ...')
        integralcheck5, error = quad(chi_235_norm, 0, 20)
        print(f"U-235 Integral result: {integralcheck5}, Estimated error: {error}")   
        integralcheck8, error = quad(chi_238_norm, 0, 20)
        print(f"U-238 Integral result: {integralcheck8}, Estimated error: {error}")
        integralcheck9, error = quad(chi_239_norm, 0, 20)
        print(f"PU-239 Integral result: {integralcheck9}, Estimated error: {error}")
        integralcheck41, error = quad(chi_241_norm, 0, 20)
        print(f"PU-241 Integral result: {integralcheck41}, Estimated error: {error}") 

        # Calculate histogram values by evaluating the function at bin centers
        hist_235 = chi_235_norm(bin_centers)
        hist_238 = chi_238_norm(bin_centers)
        hist_239 = chi_239_norm(bin_centers)
        hist_241 = chi_241_norm(bin_centers)

        print('Step 8.2: preparing power to source conversion factors ...')

        #prepare plot for spectrum
        plt.figure(dpi=300, figsize=(10, 6))

        #finish building factors
        for assembly in assembly_list:
            if assembly['r-f-d'] == 'FUEL':


                for i in range(1): #N.B. just one axial node in 2D model

                    # average assembly neutron multiplication factor (axially discretized)
                    assembly['nu'].append(np.sum(nu_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                nu_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                    # average assembly energy recoverable from fission (axially discretized)
                    assembly['erf'].append(np.sum(erf_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                erf_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                    # assembly conversion factor (axially discretized)
                    if (cycle.polarisoption == 0):
                        print("Polaris option selected: using nubar and ERF from literature assumptions")
                        assembly['F'].append(assembly['nu'][i]/(assembly['erf'][i]*C))
                    elif (cycle.polarisoption == 1):
                        print("Polaris option selected: using nubar and ERF from lattice library")
                        assembly['F'].append(assembly['nubar'][i]/(assembly['ERC'][i]*C))



                for j in range(len(bin_centers)):
                    # average assembly fission spectrum (1 spectrum per assembly)
                    assembly['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*assembly['U235'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))+
                                                hist_238[j]*(nu_238*sigma_f_238*assembly['U238'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))+
                                                hist_239[j]*(nu_239*sigma_f_239*assembly['Pu239'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))+
                                                hist_241[j]*(nu_241*sigma_f_241*assembly['Pu241'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))))
                
                print('Normalizing weighted fission spectrum ...')
                # integrate raw fission spectrum
                bin_widths = np.diff(self.bins)
                integral = np.sum(np.array(assembly['chi']) * bin_widths)
                #print('Raw weighted fission spectrum integral: ' + str(integral))
                #normalize fission spectrum
                assembly['chi'] = np.array(assembly['chi'])/integral
                #check if normalization worked
                integral= np.sum(np.array(assembly['chi'])*bin_widths)
                #print('Normalized weighted fission spectrum integral: ' + str(integral))

                # SANITY CHECK: plot fission spectrum
                if (assembly['coordinates'] in geom.source):
                    plt.step(bin_centers, assembly['chi'], label= 'Assembly ' + str(assembly['coordinates']))
                    plt.legend(fontsize=3)
                    #plt.xscale('log')
                    plt.xlabel('Energy [MeV]')
                    plt.ylabel('Probability(-)')
                    plt.title('Weighted Fission Spectrum')
        plt.grid()
        plt.savefig(os.path.join(plotpath, f'fission_spectrum_{self.step}.png'))
        plt.close()

        # Save fission spectrum to CSV
        with open(os.path.join(checkspath, f'fission_spectrum_{self.step}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Energy (MeV)'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
            writer.writerow(headers)
            for i in range(len(bin_centers)):
                row = [bin_centers[i]]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['chi'][i])
                writer.writerow(row)
                                            
        print('Step 8.3: computing source ...')
        c= 0

        # iterate on each pin to get source term
        for pin in cycle.pinpowerdata2D:
            if ([pin[1], pin[2]] in geom.source) and (pin [0] == self.step+1):
                c += 1
                print('Computing source for assembly layer n. ' + str(c))
                for assembly in assembly_list:
                    if (pin[1], pin[2]) == (assembly['coordinates'][0], assembly['coordinates'][1]):
                        if (self.truncoption == False):
                            # flip assembly to get factor from bottom to top
                            factor= np.flip(assembly['F'])
                            # extract fuel type for source definition
                            fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]
                            # multiply to get source term and overwrite  pin data
                            # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                            self.sourcepin2D.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                            break
                        elif (self.truncoption == True):
                            # flip assembly to get factor from bottom to top
                            factor= np.flip(assembly['F'])     
                            # extract fuel type for source definition
                            fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]
                   
                            if ([pin[1], pin[2]] in self.trunc_ass_X):
                                if ([pin[4],pin[5]] not in self.trunc_pin_X) and ([pin[4],pin[5]] not in self.trunc_pinsym_X):
                                    # multiply to get source term and overwrite  pin data
                                    # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                    self.sourcepin2D.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                    break
                                if ([pin[4],pin[5]] in self.trunc_pinsym_X): # if the pin is on the symmetry line, its power is divided by two
                                    # multiply to get source term and overwrite  pin data
                                    # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                    self.sourcepin2D.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                    break
                            elif ([pin[1], pin[2]] in self.trunc_ass_Y):
                                if ([pin[4],pin[5]] not in self.trunc_pin_Y) and ([pin[4],pin[5]] not in self.trunc_pinsym_Y):
                                    # multiply to get source term and overwrite  pin data
                                    # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                    self.sourcepin2D.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                    break
                                
                                if ([pin[4],pin[5]] in self.trunc_pinsym_Y): # if the pin is on the symmetry line, its power is divided by two
                                    # multiply to get source term and overwrite  pin data
                                    # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                    self.sourcepin2D.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                    break
                            else:
                                self.sourcepin2D.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                break


        # save total source rate to check file
        with open(os.path.join(checkspath, 'source_rate_pin3D.txt'), 'w', newline='') as f:
            f.write('Total source rate: ' + str(np.sum([x[7] for x in self.sourcepin2D])))

    def write_source_pin2D(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs", filepath: Union[str, Path]) -> None:
        """
        Write pin-level neutron source terms to external source file format.
        
        This method outputs the computed pin-level source data to Serpent-compatible external source
        specification files. It handles spatial binning for both point and volume sources, applies
        energy spectrum output, and implements truncation geometry transformations if specified.
        Pin-level sources provide higher spatial resolution than assembly-level approximations.
        
        Parameters
        ----------
        cycle : Cycle
            Cycle object containing polaris options and energy group information.
        geom : Geometry
            Geometry object containing core mesh, assembly pitch, pin pitch, z-coordinates,
            and mesh height information for spatial coordinate calculations.
        out : Outputs
            Outputs object for managing output directories and file paths.
        filepath : Union[str, Path]
            Path to the Serpent main input file to be updated with source strength normalization.
        
        Returns
        -------
        None
            Writes external source files to the neutron_source directory and updates the main
            Serpent input file with the total source strength.
        
        Notes
        -----
        - Creates two output files per step: external source specification (PWS.ser) and 
          updated main input file (main_*PWS.ser).
        - Supports both 'pointsource' and 'volumesource' options via self.sourceoption.
        - Point sources are defined as single coordinates (sp syntax).
        - Volume sources are defined as spatial bins (sx, sy, sz syntax) with material specification.
        - For truncation geometry, applies spatial restrictions to sources on symmetry axes.
        - Energy spectrum is output using self.bins for bin edges and computed spectrum values.
        - Normalizes source weights by total source strength for probabilistic sampling.
        - Pins on truncation symmetry axes (trunc_pinsym_X, trunc_pinsym_Y) have restricted
          volume bounds to one side of the symmetry plane.
        
        Raises
        ------
        FileNotFoundError
            If the input filepath for the main Serpent file does not exist.
        IOError
            If output directory cannot be created or files cannot be written.
        
        See Also
        --------
        build_source_pin : Pin-level source term generation.
        write_source_ass : Assembly-level source term output.
        
        Examples
        --------
        >>> source = Source(step=0, bins=np.logspace(-2, 1, 50), truncoption=False,
        ...                 assytype_to_mcmaterial={'1': 'fuel1'}, sourceoption='pointsource')
        >>> source.build_source_pin(cycle, geom, out)
        >>> source.write_source_pin(cycle, geom, out, 'path/to/main.ser')
        """


        # define output directory for source
        outpath = Path(os.path.join(out.base_dir, 'neutron_source'))
        outpath.mkdir(parents=True, exist_ok=True)

        print('Writing source term ...')
        source_sum= 0
        index= 0

        z_core_flip= np.flip(geom.z_core)

        # compute total source strength
        for pin in self.sourcepin2D:

            # sum source of neutrons
            source_sum += pin[7]

        with open(os.path.join(outpath,'LWR-10-external_source_' + str(self.step) + 'PWS.ser'), 'w') as f:
            
            for pin in self.sourcepin2D:

                index += 1

                # write source for pin
                print('Writing source for pin ...' + str(index))

                # compute pin coordinates (UPDATED)
                cA = (geom.nass + 1) / 2    # per 15 -> 8.0
                cP = (geom.npin + 1) / 2    # per 14 -> 7.5
                x_core = (pin[2] - cA) * geom.ass_pitch
                y_core = (cA - pin[1]) * geom.ass_pitch
                x_pin = x_core + (pin[5] - cP) * geom.pin_pitch
                y_pin = y_core + (cP - pin[4]) * geom.pin_pitch
                z_pin = z_core_flip[pin[3]-1]
                node_height= geom.meshheight[pin[3]-1]
                
                # compute pin boundaries
                x_min= x_pin - geom.pin_pitch/2
                x_max= x_pin + geom.pin_pitch/2
                y_min= y_pin - geom.pin_pitch/2
                y_max= y_pin + geom.pin_pitch/2
                z_min= z_pin - node_height/2
                z_max= z_pin + node_height/2

                # fuel type
                fuel_type= pin[9]
                # write neutron source 
                if self.sourceoption=='pointsource':
                
                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X):
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                if self.sourceoption=='volumesource':

                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                    
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X):
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_pin:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                        elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_pin:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                           
                f.write('1E-11 0.0\n') # insert first bin with zero value
                #insert energy spectrum specification
                print('Writing energy spectrum for pin ...' + str(index))
                for i in range(len(self.bins) - 1):
                    f.write(f"{self.bins[i+1]:.5e} {pin[8][i]:.5e}\n")
            
        with open(os.path.join(filepath), 'r') as f:
            lines = f.readlines()

        print ('Writing total source strength for normalization ...')
        with open(os.path.join(outpath, 'LWR-09-main_' + str(self.step) + 'PWS.ser'), 'w') as f:
            for i, line in enumerate(lines):
                if 'set srcrate' in line:
                    lines[i] = f'set srcrate {source_sum:.5e}\n'
                    break
            f.writelines(lines)

    # VERA related methods
    def build_source_pinVERA(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs") -> None:
        """
        Build 3D pin-level neutron source terms from VERA (MPACT) pin power data.
        """
        # define output directories
        plotpath = Path(os.path.join(out.base_dir, 'plots/pin_source'))
        plotpath.mkdir(parents=True, exist_ok=True)
        checkspath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        checkspath.mkdir(parents=True, exist_ok=True)



        print('Step 8.1: preparing data for source term ...')

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(geom.naxial,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[]}
            assembly_list.append(assembly)

        # define which assemblies are fuel and which are reflector or dummies
        for assembly in assembly_list:
            if assembly['assytype'] == 0:
                assembly['r-f-d']= 'DUMMY'
            elif assembly['assytype'] != 0:
                for axial in cycle.assyaxial:
                    if axial[1] == assembly['assytype']:
                        assembly['r-f-d']= axial[0]

        # extract burnup information only when PARCS depletion is used
        if cycle.explicitdeploption == False:
            # extract the burnup ID from exposure info
            for assembly in assembly_list:
                for assy in cycle.assyexp:
                    if assembly['coordinates'] == [assy[0], assy[1]]:
                        assembly['burnupID']= assy[2]

            # extract the burnup profile from last exposure point for each assembly
            for assembly in assembly_list:
                if assembly['burnupID'] != []:
                    print('Extracting burnup profile: ' + str(cycle.exposure[:, assembly['burnupID'] - 1, 0].flatten().tolist()))
                    assembly['buprofile'] = cycle.exposure[:, assembly['burnupID'] - 1, 0].flatten().tolist() #USER INPUT: currenly extracting specific burnup step: assemblies which have burnupID = 0 will have a burnup profile of 0
                                                                                                            #HARD CODED: the index 0 is used to indicate HFP conditions at BOC 1 with fresh fuel 

            # extract the lattice configuration of each assembly type
            for assembly in assembly_list:
                if assembly['burnupID'] != []:
                    for axialinfo in cycle.assyaxial:
                        if axialinfo[1] == assembly['assytype']:
                            assembly['latID']= axialinfo[2]
                elif assembly['burnupID'] == []: #preassign values to reflector assembly types
                    assembly['latID']= [0]*geom.naxial
                    assembly['burnupID']= 0
                    assembly['buprofile']= [0]*geom.naxial
                    assembly['bu_clos']= [0]*geom.naxial
                    assembly['nubar']= 0
                    assembly['ERC']= 0
                    assembly['U235']= [0]*geom.naxial
                    assembly['U238']= [0]*geom.naxial
                    assembly['Pu239']= [0]*geom.naxial
                    assembly['Pu241']= [0]*geom.naxial
            
            # extract the specific lattice composition: find the corresponding burnup point in lattice library and the composition
            for assembly in assembly_list:
                bp_index= -1 
                for lattice in assembly['latID']:
                    if (lattice in cycle.fuellat):
                        bp_index += 1
                        for latcomp in cycle.latcomp:
                            if (lattice == int(float(latcomp[0]))): # only fuel lattices should look for uranium and plutonium composition
                                #print(bp_index)
                                bppoint = assembly['buprofile'][bp_index]
                                closest_burnup = min(cycle.latburn, key=lambda x: abs(x - bppoint))
                                assembly['bu_clos'].append(closest_burnup)

                                if (cycle.polarisoption == 0) or (cycle.polarisoption == 1):
                                    for idx in range(0, len(latcomp), 7):
                                        if float(latcomp[idx + 1]) == closest_burnup:
                                            assembly['ERC'].append(float(latcomp[idx + 2]))
                                            assembly['U235'].append(float(latcomp[idx + 3]))
                                            assembly['U238'].append(float(latcomp[idx + 4]))
                                            assembly['Pu239'].append(float(latcomp[idx + 5]))
                                            assembly['Pu241'].append(float(latcomp[idx + 6]))
                                            assembly['nubar'].append(float(latcomp[idx + 7]))
                                            break

                                if (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                                    print('Polaris Option 2 or 3 are currently not available for pin-wise source modeling.')
                                    raise SystemExit

                    elif (lattice in cycle.refllat):
                        bp_index += 1
                        assembly['nubar'].append(0.0)
                        assembly['ERC'].append(0.0)
                        assembly['bu_clos'].append(0.0)
                        assembly['U235'].append(0)
                        assembly['U238'].append(0)
                        assembly['Pu239'].append(0)
                        assembly['Pu241'].append(0)
        
            #SANITY CHECK: plot burnup profile for each assembly
            plt.figure(dpi=300, figsize=(10, 6))
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    plt.plot(assembly['buprofile'], label=f'Assembly {assembly["coordinates"]}')
                    plt.xlabel('Axial node')
                    plt.ylabel('Burnup (MWd/kgHM)')
                    plt.title('Assembly-wise burnup profile')
                    plt.legend(fontsize=3)
                plt.grid()
            plt.savefig(plotpath / 'assembly_burnup_comparison.png')

            #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
            fig = plt.figure(dpi=300, figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    avg_burnup = np.mean(assembly['buprofile'][1:(geom.naxial-1)])
                    for i in range(1,(geom.naxial-1)):
                        # the values of the burnup profile are represented by the colormap
                        ax.bar3d(assembly['coordinates'][0], assembly['coordinates'][1], i, 1, 1, 1, shade=True, color=plt.cm.plasma(assembly['buprofile'][i]/(geom.naxial-1)), edgecolor= 'black', linewidth= 0.2)
            #create a legend next to the plot which writes the assembly location and average burnup in descending order
            # Create a legend for the plot
            legend_elements = []
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    avg_burnup = np.mean(assembly['buprofile'][1:(geom.naxial-1)])
                    legend_elements.append(f'Assembly {assembly["coordinates"]}: {avg_burnup:.2f} MWd/kgHM')

            # Add the legend to the plot
            legend_text = "\n".join(legend_elements)
            plt.figtext(0.08, 0.14, legend_text, horizontalalignment='left', fontsize=6, bbox=dict(facecolor='lightgrey', alpha=0.5))
            plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=(geom.naxial-1))), ax=ax, label='Burnup (MWd/kgHM)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            #change the orientation of the graph
            ax.view_init(elev=30, azim=45)
            #plt.title('3D Assembly-wise burnup profile')
            plt.savefig(plotpath / f'3D_assembly_burnup_comparison_{self.step}.png', bbox_inches='tight')

            # save the data for these assemblies to a csv file
            with open(checkspath / f'burnup_profile_{self.step}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
                writer.writerow(headers)
                for i in range(geom.naxial):
                    row = [i]
                    for assembly in assembly_list:
                        if assembly['coordinates'] in geom.source:
                            row.append(assembly['buprofile'][i])
                    writer.writerow(row)

            #SANITY CHECK: plot U235 composition for each assembly
            plt.figure(dpi=300, figsize=(10, 6))
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    plt.plot(assembly['U235'], label=f'Assembly {assembly["coordinates"]}')
                    plt.xlabel('Axial node')
                    plt.ylabel('U235 Atomic Density')
                    plt.title('Assembly-wise U235 composition')
                    plt.legend(fontsize=3)
                plt.grid()
            plt.savefig(plotpath / 'assembly_u235_comparison.png')

            #SANITY CHECK: plot U238 composition for each assembly
            plt.figure(dpi=300, figsize=(10, 6))
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    plt.plot(assembly['U238'], label=f'Assembly {assembly["coordinates"]}')
                    plt.xlabel('Axial node')
                    plt.ylabel('U238 Atomic Density')
                    plt.title('Assembly-wise U238 composition')
                    plt.legend(fontsize=3)
                plt.grid()
            plt.savefig(plotpath / 'assembly_u238_comparison.png')

            #SANITY CHECK: plot PU239 composition for each assembly
            plt.figure(dpi=300, figsize=(10, 6))
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    plt.plot(assembly['Pu239'], label=f'Assembly {assembly["coordinates"]}')
                    plt.xlabel('Axial node')
                    plt.ylabel('Pu239 Atomic Density')
                    plt.title('Assembly-wise PU239 composition')
                    plt.legend(fontsize=3)
                plt.grid()
            plt.savefig(plotpath / 'assembly_pu239_comparison.png')

            #SANITY CHECK: plot PU241 composition for each assembly
            plt.figure(dpi=300, figsize=(10, 6))
            for assembly in assembly_list:
                if assembly['coordinates'] in geom.source:
                    plt.plot(assembly['Pu241'], label=f'Assembly {assembly["coordinates"]}')
                    plt.xlabel('Axial node')
                    plt.ylabel('Pu241 Atomic Density')
                    plt.title('Assembly-wise PU241 composition')
                    plt.legend(fontsize=3)
                plt.grid()
            plt.savefig(plotpath / 'assembly_pu241_comparison.png')

        # extract the factors necessary for the source term

        # microscopic fission cross-sections (barns)
        sigma_f_235 = 566.0
        sigma_f_238 = 1.5
        sigma_f_239 = 781.0
        sigma_f_241 = 1060.0 

        # neutron yield per fission
        nu_235 = 2.430 # MCNP-DATA  # ENDFBVIII.0 = 2.414000 
        nu_238 = 2.810 # MCNP-DATA  # ENDFBVIII.0 = 2.611820
        nu_239 = 2.871 # MCNP-DATA  # ENDFBVIII.0 = 2.868503 
        nu_241 = 2.969 # MCNP-DATA  # ENDFBVIII.0 = 2.929100

        # energy recoverable from fission
        erf_235 = 201.7
        erf_238 = 205.0
        erf_239 = 210.0
        erf_241 = 212.4

        # Constants
        E = np.linspace(0, 20, 1000)
        C = 1.6019e-13 # MeV to Joules

        # Define fission spectrum functions (Watt and Maxwell)
        def chi_235(E):
            return np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238(E):
            return np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239(E):
            return np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241(E):
            return np.sqrt(E) * np.exp(-E / 1.360)

        # Energy bins - defined center points
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])  # Midpoints of bins

        # integrate raw fission spectrum
        print('computing spectrum integral for normalization ...')
        integral5, error = quad(chi_235, 0, 20)
        print(f"U-235 Integral result: {integral5}, Estimated error: {error}")   
        integral8, error = quad(chi_238, 0, 20)
        print(f"U-238 Integral result: {integral8}, Estimated error: {error}")
        integral9, error = quad(chi_239, 0, 20)
        print(f"PU-239 Integral result: {integral9}, Estimated error: {error}")
        integral41, error = quad(chi_241, 0, 20)
        print(f"PU-241 Integral result: {integral41}, Estimated error: {error}")    

        # Define fission spectrum functions
        def chi_235_norm(E):
            return 1/integral5 * np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238_norm(E):
            return 1/integral8* np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239_norm(E):
            return 1/integral9* np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241_norm(E):
            return 1/integral41* np.sqrt(E) * np.exp(-E / 1.360)

        # integrate raw fission spectrum
        print('checking new integration ...')
        integralcheck5, error = quad(chi_235_norm, 0, 20)
        print(f"U-235 Integral result: {integralcheck5}, Estimated error: {error}")   
        integralcheck8, error = quad(chi_238_norm, 0, 20)
        print(f"U-238 Integral result: {integralcheck8}, Estimated error: {error}")
        integralcheck9, error = quad(chi_239_norm, 0, 20)
        print(f"PU-239 Integral result: {integralcheck9}, Estimated error: {error}")
        integralcheck41, error = quad(chi_241_norm, 0, 20)
        print(f"PU-241 Integral result: {integralcheck41}, Estimated error: {error}") 

        # Calculate histogram values by evaluating the function at bin centers
        hist_235 = chi_235_norm(bin_centers)
        hist_238 = chi_238_norm(bin_centers)
        hist_239 = chi_239_norm(bin_centers)
        hist_241 = chi_241_norm(bin_centers)

        print('Step 8.2: preparing power to source conversion factors ...')

        #prepare plot for spectrum
        plt.figure(dpi=300, figsize=(10, 6))

        #finish building factors
        for assembly in assembly_list:
            
            if assembly['r-f-d'] == 'FUEL':

                # first node is a reflector
                assembly['nu'].append(0.0)
                assembly['erf'].append(0.0)
                assembly['F'].append(0.0)

                # distinguish the explicit depletion option from the one reading isotope information from 
                if cycle.explicitdeploption == False:

                    for i in range(1,(geom.naxial-1)):
                        # average assembly neutron multiplication factor (axially discretized)
                        assembly['nu'].append(np.sum(nu_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                            nu_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                            nu_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                            nu_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                        # average assembly energy recoverable from fission (axially discretized)
                        assembly['erf'].append(np.sum(erf_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                            erf_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                            erf_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                            erf_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                        # assembly conversion factor (axially discretized)
                        if (cycle.polarisoption == 0):
                            print("Polaris option selected: using nubar and ERF from literature assumptions")
                            assembly['F'].append(assembly['nu'][i]/(assembly['erf'][i]*C))
                        elif (cycle.polarisoption == 1):
                            print("Polaris option selected: using nubar and ERF from lattice library")
                            assembly['F'].append(assembly['nubar'][i]/(assembly['ERC'][i]*C))

                    for j in range(len(bin_centers)):
                        # average assembly fission spectrum (1 spectrum per assembly)
                        assembly['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_238[j]*(nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_239[j]*(nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))+
                                                    hist_241[j]*(nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)]))/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][1:(geom.naxial-1)])+nu_238*sigma_f_238*np.average(assembly['U238'][1:(geom.naxial-1)])+nu_239*sigma_f_239*np.average(assembly['Pu239'][1:(geom.naxial-1)])+nu_241*sigma_f_241*np.average(assembly['Pu241'][1:(geom.naxial-1)])))))
                    
                    print('Normalizing weighted fission spectrum ...')
                    # integrate raw fission spectrum
                    bin_widths = np.diff(self.bins)
                    integral = np.sum(np.array(assembly['chi']) * bin_widths)
                    #print('Raw weighted fission spectrum integral: ' + str(integral))
                    #normalize fission spectrum
                    assembly['chi'] = np.array(assembly['chi'])/integral
                    #check if normalization worked
                    integral= np.sum(np.array(assembly['chi'])*bin_widths)
                    #print('Normalized weighted fission spectrum integral: ' + str(integral))

                # last node is a reflector
                assembly['nu'].append(0.0)
                assembly['erf'].append(0.0)
                assembly['F'].append(0.0)

                if cycle.explicitdeploption == False: # you only plot the spectrum if explicit depletion is not selected, because it's on an assembly level
                    # SANITY CHECK: plot fission spectrum
                    if (assembly['coordinates'] in geom.source):
                        plt.step(bin_centers, assembly['chi'], label= 'Assembly ' + str(assembly['coordinates']))
                        plt.legend(fontsize=3)
                        #plt.xscale('log')
                        plt.xlabel('Energy [MeV]')
                        plt.ylabel('Probability(-)')
                        plt.title('Weighted Fission Spectrum')
                        plt.grid()
                        plt.savefig(plotpath / f'fission_spectrum_{self.step}.png')
        
        if cycle.explicitdeploption == False:
            # Save fission spectrum to CSV
            with open(checkspath / f'fission_spectrum_{self.step}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = ['Energy (MeV)'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
                writer.writerow(headers)
                for i in range(len(bin_centers)):
                    row = [bin_centers[i]]
                    for assembly in assembly_list:
                        if assembly['coordinates'] in geom.source:
                            row.append(assembly['chi'][i])
                    writer.writerow(row)
                                            
        print('Step 8.3: computing source ...')
        c= 0

        if cycle.interpoption: # WARNING: this option is not available for VERA
            print('Interpolation Option Activated ...')
            # iterate on each pin to get source term
            for pin in cycle.pinpowerdatainterp:
                if ([pin[1], pin[2]] in geom.source) and (pin [0] == self.step): # modified to extract information at any step
                    c += 1
                    print('Computing source for assembly layer n. ' + str(c))
                    for assembly in assembly_list:
                        if (pin[1], pin[2]) == (assembly['coordinates'][0], assembly['coordinates'][1]):


                            if (self.truncoption == False):

                                # flip assembly to get factor from bottom to top
                                factor= np.flip(assembly['F'])
                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                                # multiply to get source term and overwrite  pin data
                                # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                
                                if cycle.explicitdeploption==False:
                                    self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                elif cycle.explicitdeploption==True:
                                    self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], pin['chi'], fueltype))

                                break

                            elif (self.truncoption == True):

                                # flip assembly to get factor from bottom to top
                                factor= np.flip(assembly['F'])     

                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                                if ([pin[1], pin[2]] in self.trunc_ass_X):

                                    if ([pin[4],pin[5]] not in self.trunc_pin_X) and ([pin[4],pin[5]] not in self.trunc_pinsym_X):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                        break

                                    if ([pin[4],pin[5]] in self.trunc_pinsym_X): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                        break


                                elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                                    if ([pin[4],pin[5]] not in self.trunc_pin_Y) and ([pin[4],pin[5]] not in self.trunc_pinsym_Y):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                        break
                                    
                                    if ([pin[4],pin[5]] in self.trunc_pinsym_Y): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6]/2, pin[6]*factor[pin[3]-1]/2, assembly['chi'], fueltype))
                                        break

                                else:
                                    self.sourcepinVERA.append((pin[0], pin[1], pin[2], pin[3], pin[4], pin[5], pin[6], pin[6]*factor[pin[3]-1], assembly['chi'], fueltype))
                                    break

        else:
            # iterate on each pin to get source term
            for pin in cycle.pinpowerdataVERA:
                if ([pin[2], pin[3]] in geom.source) and (pin [0] == self.step): # modified to extract information at any step
                    c += 1
                    print('Computing source for assembly layer n. ' + str(c))
                    for assembly in assembly_list:
                        if (pin[2], pin[3]) == (assembly['coordinates'][0], assembly['coordinates'][1]):

                            # read the isotope information from the pin dictionary
                            # GUIDE: u235 = pin[11]
                            #        u238 = pin[12]
                            #        pu239 = pin[13]
                            #        pu241 = pin[14]

                            if cycle.explicitdeploption == True:

                                # create a dictionary to store relevant information for pinwise source specification, this is overwritten for each pin slice
                                frod= {'nu':[], 'erf':[], 'F':[], 'chi':[]}

                                # NOTE: using isotope information from pin dictionary and literature values for nubar and ERF
                                # pin neutron multiplication factor (axially discretized)
                                frod['nu']= np.sum(nu_235*(sigma_f_235*pin[11])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                        nu_238*(sigma_f_238*pin[12])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                        nu_239*(sigma_f_239*pin[13])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                        nu_241*(sigma_f_241*pin[14])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14])))
                                # pin energy recoverable from fission (axially discretized)
                                frod['erf']= np.sum(erf_235*(sigma_f_235*pin[11])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                        erf_238*(sigma_f_238*pin[12])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                        erf_239*(sigma_f_239*pin[13])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                        erf_241*(sigma_f_241*pin[14])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14])))
                                    
                                # pin conversion factor (axially discretized)
                                frod['F'] = frod['nu']/(frod['erf']*C)
                                #print('Pinwise conversion factor at node ' + str(i) + ': ' + str(frod['F'][i-1]))

                                for j in range(len(bin_centers)):
                                    # pinwise fission spectrum
                                    frod['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*pin[11])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))+
                                                       hist_238[j]*(nu_238*sigma_f_238*pin[12])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))+
                                                       hist_239[j]*(nu_239*sigma_f_239*pin[13])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))+
                                                       hist_241[j]*(nu_241*sigma_f_241*pin[14])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))))
                                    
                                print('Normalizing weighted fission spectrum ...')
                                # integrate raw fission spectrum
                                bin_widths = np.diff(self.bins)
                                integral = np.sum(np.array(frod['chi'])*bin_widths)
                                #print('Raw weighted fission spectrum integral: ' + str(integral))
                                #normalize fission spectrum
                                frod['chi'] = np.array(frod['chi'])/integral
                                #check if normalization worked
                                integral= np.sum(np.array(frod['chi'])*bin_widths)
                                #print('Normalized weighted fission spectrum integral: ' + str(integral))
                                                    
                            if (self.truncoption == False):

                                # only if PARCS burnup information have to be used the factor needs to be flipped, because it is defined assembly-wise
                                if cycle.explicitdeploption == False:
                                    # flip assembly to get factor from bottom to top
                                    factor= np.flip(assembly['F'])
                                elif cycle.explicitdeploption == True:
                                    factor = frod['F']

                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                                # multiply to get source term and overwrite  pin data
                                # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                if cycle.explicitdeploption==False:
                                    self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[pin[4]+1], assembly['chi'], fueltype))
                                elif cycle.explicitdeploption==True:
                                    self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))

                                break

                            elif (self.truncoption == True):

                                # only if PARCS burnup information have to be used the factor needs to be flipped
                                if cycle.explicitdeploption == False:
                                    # flip assembly to get factor from bottom to top
                                    factor= np.flip(assembly['F'])
                                elif cycle.explicitdeploption == True:
                                    # The factor is rewritten for each pin axial level when explicit depletion is selected
                                    factor = frod['F']    

                                # extract fuel type for source definition
                                fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                                if ([pin[2], pin[3]] in self.trunc_ass_X):

                                    if ([pin[5],pin[6]] not in self.trunc_pin_X) and ([pin[5],pin[6]] not in self.trunc_pinsym_X):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        if cycle.explicitdeploption==False:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[pin[4]+1], assembly['chi'], fueltype))
                                        elif cycle.explicitdeploption==True:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))
                                           
                                        break

                                    if ([pin[5],pin[6]] in self.trunc_pinsym_X): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        if cycle.explicitdeploption==False:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[pin[4]+1]/2, assembly['chi'], fueltype))
                                        elif cycle.explicitdeploption==True:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor/2, frod['chi'], fueltype))

                                        break


                                elif ([pin[2], pin[3]] in self.trunc_ass_Y):

                                    if ([pin[5],pin[6]] not in self.trunc_pin_Y) and ([pin[5],pin[6]] not in self.trunc_pinsym_Y):
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        if cycle.explicitdeploption==False:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[pin[4]+1], assembly['chi'], fueltype))
                                        elif cycle.explicitdeploption==True:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))

                                        break
                                    
                                    if ([pin[5],pin[6]] in self.trunc_pinsym_Y): # if the pin is on the symmetry line, its power is divided by two
                                        # multiply to get source term and overwrite  pin data
                                        # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (assembly fission spectrum)
                                        if cycle.explicitdeploption==False:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[pin[4]+1]/2, assembly['chi'], fueltype))
                                        elif cycle.explicitdeploption==True:
                                            self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor/2, frod['chi'], fueltype))
                                        break

                                else:
                                    if cycle.explicitdeploption==False:
                                        self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[pin[4]+1], assembly['chi'], fueltype))
                                    elif cycle.explicitdeploption==True:
                                        self.sourcepinVERA.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))
                                    break


        # save total source rate to check file
        with open(checkspath / 'source_rate_pin3D.txt', 'w', newline='') as f:
            f.write('Total source rate: ' + str(np.sum([x[7] for x in self.sourcepinVERA])))

        # save assembly list structure
        self.F= assembly_list           

    def build_source_pinVERA2D(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs") -> None:
        """
        Build 2D pin-level neutron source terms from VERA (MPACT) pin power data.
        """
        # define output directories
        plotpath = Path(os.path.join(out.base_dir, 'plots/pin_source'))
        plotpath.mkdir(parents=True, exist_ok=True)
        checkspath = Path(os.path.join(out.base_dir, 'sanity_checks/pin_source'))
        checkspath.mkdir(parents=True, exist_ok=True)


        
        # This is a hard-coded version that takes the fuel composition and the spectrum information from Polaris/PARCS and changes only the power prediction
        # It works only when the VERA output is in 2D

        print('Step 8.1: preparing data for source term ...')

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(geom.naxial,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[]}
            assembly_list.append(assembly)

        # define which assemblies are fuel and which are reflector or dummies
        for assembly in assembly_list:
            if assembly['assytype'] == 0:
                assembly['r-f-d']= 'DUMMY'
            elif assembly['assytype'] != 0:
                for axial in cycle.assyaxial:
                    if axial[1] == assembly['assytype']:
                        assembly['r-f-d']= axial[0]

        # extract the burnup ID from exposure info
        for assembly in assembly_list:
            for assy in cycle.assyexp:
                if assembly['coordinates'] == [assy[0], assy[1]]:
                    assembly['burnupID']= assy[2]

        # extract the burnup profile from last exposure point for each assembly
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                assembly['buprofile'] = cycle.exposure[:, assembly['burnupID'] - 1, 0].flatten().tolist() #USER INPUT: currenly extracting specific burnup step: assemblies which have burnupID = 0 will have a burnup profile of 0
                                                                                                        #HARD CODED: the index 0 is used to indicate HFP conditions at BOC 1 with fresh fuel 
        # extract the lattice configuration of each assembly type
        for assembly in assembly_list:
            if assembly['burnupID'] != []:
                for axialinfo in cycle.assyaxial:
                    if axialinfo[1] == assembly['assytype']:
                        assembly['latID']= axialinfo[2] 
            elif assembly['burnupID'] == []: #preassign values to reflector assembly types
                assembly['latID']= [0]*geom.naxial
                assembly['burnupID']= 0
                assembly['buprofile']= [0]*geom.naxial
                assembly['bu_clos']= [0]*geom.naxial
                assembly['ERC']= 0
                assembly['nubar']= 0
                assembly['U235']= [0]*geom.naxial
                assembly['U238']= [0]*geom.naxial
                assembly['Pu239']= [0]*geom.naxial
                assembly['Pu241']= [0]*geom.naxial
        
        # extract the specific lattice composition: find the corresponding burnup point in lattice library and the composition
        for assembly in assembly_list:
            bp_index= -1 
            for lattice in assembly['latID']:
                if (lattice in cycle.fuellat):
                    bp_index += 1
                    for latcomp in cycle.latcomp:
                        if (lattice == int(float(latcomp[0]))): # only fuel lattices should look for uranium and plutonium composition
                            bppoint = assembly['buprofile'][bp_index]
                            closest_burnup = min(cycle.latburn, key=lambda x: abs(x - bppoint))
                            assembly['bu_clos'].append(closest_burnup)
                            for idx in range(0, len(latcomp), 7):
                                if float(latcomp[idx + 1]) == closest_burnup:
                                    assembly['ERC']= float(latcomp[idx + 2])
                                    assembly['U235'].append(float(latcomp[idx + 3]))
                                    assembly['U238'].append(float(latcomp[idx + 4]))
                                    assembly['Pu239'].append(float(latcomp[idx + 5]))
                                    assembly['Pu241'].append(float(latcomp[idx + 6]))
                                    assembly['nubar']= float(latcomp[idx + 7])
                                    break
                elif (lattice in cycle.refllat):
                    bp_index += 1
                    assembly['nubar']= 0
                    assembly['ERC']= 0
                    assembly['bu_clos'].append(0.0)
                    assembly['U235'].append(0)
                    assembly['U238'].append(0)
                    assembly['Pu239'].append(0)
                    assembly['Pu241'].append(0)
        
        #SANITY CHECK: plot burnup profile for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['buprofile'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Burnup (MWd/kgHM)')
                plt.title('Assembly-wise burnup profile')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / 'assembly_burnup_comparison.png')

        #make a 3D checkerboard plot for the burnup profile, considering only the quarter checkerboard that is indicated in asso
        fig = plt.figure(dpi=300, figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = assembly['buprofile'][0] # HARD CODED: for the VERA 2D case there is just one node!
                for i in range(1,(geom.naxial-1)):
                    # the values of the burnup profile are represented by the colormap
                    ax.bar3d(assembly['coordinates'][0], assembly['coordinates'][1], i, 1, 1, 1, shade=True, color=plt.cm.bwr(assembly['buprofile'][i]/20), edgecolor= 'black', linewidth= 0.2)
        #create a legend next to the plot which writes the assembly location and average burnup in descending order
        # Create a legend for the plot
        legend_elements = []
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                avg_burnup = assembly['buprofile'][0] # HARD CODED: for the VERA 2D case there is just one node!
                legend_elements.append(f'Assembly {assembly["coordinates"]}: {avg_burnup:.2f} MWd/kgHM')

        # Add the legend to the plot
        legend_text = "\n".join(legend_elements)
        plt.figtext(0.01, 0.5, legend_text, horizontalalignment='left', fontsize=6, bbox=dict(facecolor='white', alpha=0.5))
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr, norm=plt.Normalize(vmin=0, vmax=20)), ax=ax, label='Burnup (MWd/kgHM)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #change the orientation of the graph
        ax.view_init(elev=30, azim=45)
        plt.title('3D Assembly-wise burnup profile')
        plt.savefig(plotpath / '3D_assembly_burnup_comparison.png')

        # save the data for these assemblies to a csv file
        with open(checkspath / f'burnup_profile_{self.step}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Axial Node'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
            writer.writerow(headers)
            for i in range(geom.naxial):
                row = [i]
                for assembly in assembly_list:
                    if assembly['coordinates'] in geom.source:
                        row.append(assembly['buprofile'][i])
                writer.writerow(row)

        #SANITY CHECK: plot U235 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U235'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U235 Atomic Density')
                plt.title('Assembly-wise U235 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / 'assembly_u235_comparison.png')

        #SANITY CHECK: plot U238 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['U238'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('U238 Atomic Density')
                plt.title('Assembly-wise U238 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / 'assembly_u238_comparison.png')

        #SANITY CHECK: plot PU239 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu239'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu239 Atomic Density')
                plt.title('Assembly-wise PU239 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / 'assembly_pu239_comparison.png')

        #SANITY CHECK: plot PU241 composition for each assembly
        plt.figure(dpi=300, figsize=(10, 6))
        for assembly in assembly_list:
            if assembly['coordinates'] in geom.source:
                plt.plot(assembly['Pu241'], label=f'Assembly {assembly["coordinates"]}')
                plt.xlabel('Axial node')
                plt.ylabel('Pu241 Atomic Density')
                plt.title('Assembly-wise PU241 composition')
                plt.legend(fontsize=3)
            plt.grid()
        plt.savefig(plotpath / 'assembly_pu241_comparison.png')

        # extract the factors necessary for the source term

        # microscopic fission cross-sections (barns)
        sigma_f_235 = 566.0
        sigma_f_238 = 1.5
        sigma_f_239 = 781.0
        sigma_f_241 = 1060.0 

        # neutron yield per fission
        nu_235 = 2.430
        nu_238 = 2.810
        nu_239 = 2.871
        nu_241 = 2.969

        # energy recoverable from fission
        erf_235 = 201.7
        erf_238 = 205.0
        erf_239 = 210.0
        erf_241 = 212.4

        # Constants
        E = np.linspace(0, 20, 1000)
        C = 1.6019e-13 # MeV to Joules

        # Define fission spectrum functions
        def chi_235(E):
            return np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238(E):
            return np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239(E):
            return np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241(E):
            return np.sqrt(E) * np.exp(-E / 1.360)

        # Energy bins - defined center points
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])  # Midpoints of bins

        # integrate raw fission spectrum
        print('computing spectrum integral for normalization ...')
        integral5, error = quad(chi_235, 0, 20)
        print(f"U-235 Integral result: {integral5}, Estimated error: {error}")   
        integral8, error = quad(chi_238, 0, 20)
        print(f"U-238 Integral result: {integral8}, Estimated error: {error}")
        integral9, error = quad(chi_239, 0, 20)
        print(f"PU-239 Integral result: {integral9}, Estimated error: {error}")
        integral41, error = quad(chi_241, 0, 20)
        print(f"PU-241 Integral result: {integral41}, Estimated error: {error}")    

        # Define fission spectrum functions
        def chi_235_norm(E):
            return 1/integral5 * np.exp(-E / 0.988) * np.sinh(np.sqrt(E * 2.249))

        def chi_238_norm(E):
            return 1/integral8* np.exp(-E / 0.920) * np.sinh(np.sqrt(E * 3.121))

        def chi_239_norm(E):
            return 1/integral9* np.exp(-E / 0.966) * np.sinh(np.sqrt(E * 2.842))

        def chi_241_norm(E):
            return 1/integral41* np.sqrt(E) * np.exp(-E / 1.360)

        # integrate raw fission spectrum
        print('checking new integration ...')
        integralcheck5, error = quad(chi_235_norm, 0, 20)
        print(f"U-235 Integral result: {integralcheck5}, Estimated error: {error}")   
        integralcheck8, error = quad(chi_238_norm, 0, 20)
        print(f"U-238 Integral result: {integralcheck8}, Estimated error: {error}")
        integralcheck9, error = quad(chi_239_norm, 0, 20)
        print(f"PU-239 Integral result: {integralcheck9}, Estimated error: {error}")
        integralcheck41, error = quad(chi_241_norm, 0, 20)
        print(f"PU-241 Integral result: {integralcheck41}, Estimated error: {error}") 

        # Calculate histogram values by evaluating the function at bin centers
        hist_235 = chi_235_norm(bin_centers)
        hist_238 = chi_238_norm(bin_centers)
        hist_239 = chi_239_norm(bin_centers)
        hist_241 = chi_241_norm(bin_centers)

        print('Step 8.2: preparing power to source conversion factors ...')

        #prepare plot for spectrum
        plt.figure(dpi=300, figsize=(10, 6))

        #finish building factors
        for assembly in assembly_list:
            if assembly['r-f-d'] == 'FUEL':

                # HARD CODED: for the VERA 2D case there is just one node!
                # # first node is a reflector
                # assembly['nu'].append(0.0)
                # assembly['erf'].append(0.0)
                # assembly['F'].append(0.0)

                if cycle.explicitdeploption == False: # with explicit depletion the spectrum/factor are built pin-wise from the VERA isotopics, so skip the assembly-level build
                    for i in range(1): # HARD CODED: for the VERA 2D case there is just one node!

                        # average assembly neutron multiplication factor (axially discretized)
                        assembly['nu'].append(np.sum(nu_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                    nu_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                    nu_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                    nu_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                        # average assembly energy recoverable from fission (axially discretized)
                        assembly['erf'].append(np.sum(erf_235*(sigma_f_235*assembly['U235'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                    erf_238*(sigma_f_238*assembly['U238'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                    erf_239*(sigma_f_239*assembly['Pu239'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))+
                                                    erf_241*(sigma_f_241*assembly['Pu241'][i])/(np.sum(sigma_f_235*assembly['U235'][i]+sigma_f_238*assembly['U238'][i]+sigma_f_239*assembly['Pu239'][i]+sigma_f_241*assembly['Pu241'][i]))))
                        # assembly conversion factor (axially discretized)
                        assembly['F'].append(assembly['nu'][i]/(assembly['erf'][i]*C))

                    for j in range(len(bin_centers)):
                        # average assembly fission spectrum (1 spectrum per assembly)
                        assembly['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*assembly['U235'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))+
                                                    hist_238[j]*(nu_238*sigma_f_238*assembly['U238'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))+
                                                    hist_239[j]*(nu_239*sigma_f_239*assembly['Pu239'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))+
                                                    hist_241[j]*(nu_241*sigma_f_241*assembly['Pu241'][0])/(np.sum(nu_235*sigma_f_235*np.average(assembly['U235'][0])+nu_238*sigma_f_238*np.average(assembly['U238'][0])+nu_239*sigma_f_239*np.average(assembly['Pu239'][0])+nu_241*sigma_f_241*np.average(assembly['Pu241'][0])))))
                
                    print('Normalizing weighted fission spectrum ...')
                    # integrate raw fission spectrum
                    bin_widths = np.diff(self.bins)
                    integral = np.sum(np.array(assembly['chi']) * bin_widths)
                    #print('Raw weighted fission spectrum integral: ' + str(integral))
                    #normalize fission spectrum
                    assembly['chi'] = np.array(assembly['chi'])/integral
                    #check if normalization worked
                    integral= np.sum(np.array(assembly['chi'])*bin_widths)
                    #print('Normalized weighted fission spectrum integral: ' + str(integral))

                    # # HARD CODED: for the VERA 2D case there is just one node!
                    # last node is a reflector
                    # assembly['nu'].append(0.0)
                    # assembly['erf'].append(0.0)
                    # assembly['F'].append(0.0)

                    # SANITY CHECK: plot fission spectrum
                    if (assembly['coordinates'] in geom.source):
                        plt.step(bin_centers, assembly['chi'], label= 'Assembly ' + str(assembly['coordinates']))
                        plt.legend(fontsize=3)
                        #plt.xscale('log')
                        plt.xlabel('Energy [MeV]')
                        plt.ylabel('Probability(-)')
                        plt.title('Weighted Fission Spectrum')
        plt.grid()
        plt.savefig(plotpath / f'fission_spectrum_{self.step}.png')

        if cycle.explicitdeploption == False: # the assembly-level spectrum CSV is meaningless under explicit (pin-wise) depletion
            # Save fission spectrum to CSV
            with open(checkspath / f'fission_spectrum_{self.step}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = ['Energy (MeV)'] + [f'Assembly {assembly["coordinates"]}' for assembly in assembly_list if assembly['coordinates'] in geom.source]
                writer.writerow(headers)
                for i in range(len(bin_centers)):
                    row = [bin_centers[i]]
                    for assembly in assembly_list:
                        if assembly['coordinates'] in geom.source:
                            row.append(assembly['chi'][i])
                    writer.writerow(row)
                                         
        print('Step 8.3: computing source ...')
        c= 0

        # iterate on each pin to get source term
        for pin in cycle.pinpowerdataVERA2D:
            if ([pin[2], pin[3]] in geom.source) and (pin [0] == 0): # HFP selection
                c += 1
                print('Computing source for assembly layer n. ' + str(c))
                for assembly in assembly_list:
                    if (pin[2], pin[3]) == (assembly['coordinates'][0], assembly['coordinates'][1]):

                        # read the isotope information from the pin dictionary
                        # GUIDE: u235 = pin[11]
                        #        u238 = pin[12]
                        #        pu239 = pin[13]
                        #        pu241 = pin[14]

                        if cycle.explicitdeploption == True:

                            # create a dictionary to store relevant information for pinwise source specification, this is overwritten for each pin slice
                            frod= {'nu':[], 'erf':[], 'F':[], 'chi':[]}

                            # NOTE: using isotope information from pin dictionary and literature values for nubar and ERF
                            # pin neutron multiplication factor
                            frod['nu']= np.sum(nu_235*(sigma_f_235*pin[11])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                    nu_238*(sigma_f_238*pin[12])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                    nu_239*(sigma_f_239*pin[13])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                    nu_241*(sigma_f_241*pin[14])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14])))
                            # pin energy recoverable from fission
                            frod['erf']= np.sum(erf_235*(sigma_f_235*pin[11])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                    erf_238*(sigma_f_238*pin[12])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                    erf_239*(sigma_f_239*pin[13])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14]))+
                                                    erf_241*(sigma_f_241*pin[14])/(np.sum(sigma_f_235*pin[11]+sigma_f_238*pin[12]+sigma_f_239*pin[13]+sigma_f_241*pin[14])))

                            # pin conversion factor
                            frod['F'] = frod['nu']/(frod['erf']*C)

                            for j in range(len(bin_centers)):
                                # pinwise fission spectrum
                                frod['chi'].append(np.sum(hist_235[j]*(nu_235*sigma_f_235*pin[11])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))+
                                                   hist_238[j]*(nu_238*sigma_f_238*pin[12])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))+
                                                   hist_239[j]*(nu_239*sigma_f_239*pin[13])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))+
                                                   hist_241[j]*(nu_241*sigma_f_241*pin[14])/(np.sum(nu_235*sigma_f_235*pin[11]+nu_238*sigma_f_238*pin[12]+nu_239*sigma_f_239*pin[13]+nu_241*sigma_f_241*pin[14]))))

                            print('Normalizing weighted fission spectrum ...')
                            # integrate raw fission spectrum
                            bin_widths = np.diff(self.bins)
                            integral = np.sum(np.array(frod['chi'])*bin_widths)
                            #normalize fission spectrum
                            frod['chi'] = np.array(frod['chi'])/integral
                            #check if normalization worked
                            integral= np.sum(np.array(frod['chi'])*bin_widths)

                        # only if PARCS/Polaris burnup information have to be used the factor needs to be flipped, because it is defined assembly-wise
                        if cycle.explicitdeploption == False:
                            # flip assembly to get factor from bottom to top
                            factor= np.flip(assembly['F'])
                            print(f'Factor for assembly {assembly["coordinates"]}, pin=({pin[5]},{pin[6]}): {factor[0]}')
                        elif cycle.explicitdeploption == True:
                            # the factor is rewritten for each pin when explicit depletion is selected
                            factor = frod['F']
                            print(f'Factor for assembly {assembly["coordinates"]}, pin=({pin[5]},{pin[6]}): {factor}')
                        print(f'Power for assembly {assembly["coordinates"]}, pin=({pin[5]},{pin[6]}): {pin[9]}')

                        # extract fuel type for source definition
                        fueltype = self.assytype_to_mcmaterial[str(assembly['assytype'])]

                        if (self.truncoption == False):
                            # case number, index i (assembly x), index j (assembly y), index k (assembly z), x_index (pin x), y_index (pin y), value (pin power), source term (pin power*factor), fission spectrum (fission spectrum)
                            if cycle.explicitdeploption == False:
                                self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[0], assembly['chi'], fueltype))
                            elif cycle.explicitdeploption == True:
                                self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))

                        elif (self.truncoption == True):

                            if ([pin[2], pin[3]] in self.trunc_ass_X):

                                if ([pin[5],pin[6]] not in self.trunc_pin_X) and ([pin[5],pin[6]] not in self.trunc_pinsym_X):
                                    # multiply to get source term and overwrite  pin data
                                    if cycle.explicitdeploption == False:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[0], assembly['chi'], fueltype))
                                    elif cycle.explicitdeploption == True:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))
                                    break

                                if ([pin[5],pin[6]] in self.trunc_pinsym_X): # if the pin is on the symmetry line, its power is divided by two
                                    if cycle.explicitdeploption == False:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9]/2, pin[9]*factor[0]/2, assembly['chi'], fueltype))
                                    elif cycle.explicitdeploption == True:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9]/2, pin[9]*factor/2, frod['chi'], fueltype))
                                    break


                            elif ([pin[2], pin[3]] in self.trunc_ass_Y):

                                if ([pin[5],pin[6]] not in self.trunc_pin_Y) and ([pin[5],pin[6]] not in self.trunc_pinsym_Y):
                                    # multiply to get source term and overwrite  pin data
                                    if cycle.explicitdeploption == False:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[0], assembly['chi'], fueltype))
                                    elif cycle.explicitdeploption == True:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))
                                    break

                                if ([pin[5],pin[6]] in self.trunc_pinsym_Y): # if the pin is on the symmetry line, its power is divided by two
                                    if cycle.explicitdeploption == False:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9]/2, pin[9]*factor[0]/2, assembly['chi'], fueltype))
                                    elif cycle.explicitdeploption == True:
                                        self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9]/2, pin[9]*factor/2, frod['chi'], fueltype))
                                    break

                            else:
                                if cycle.explicitdeploption == False:
                                    self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor[0], assembly['chi'], fueltype))
                                elif cycle.explicitdeploption == True:
                                    self.sourcepinVERA2D.append((pin[0], pin[2], pin[3], pin[4], pin[5], pin[6], pin[9], pin[9]*factor, frod['chi'], fueltype))
                                break


        # save assembly list structure
        self.F= assembly_list       

        # save total source rate to check file
        with open(checkspath / 'source_rate_pin2D.txt', 'w', newline='') as f:
            f.write('Total source rate: ' + str(np.sum([x[7] for x in self.sourcepinVERA2D])))

    def write_source_pinVERA(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs", filepath: Union[str, Path]) -> None:
        """
        Write 3D VERA pin-level neutron source terms to Serpent external source format.
        """
        # define output directory for source
        outpath = Path(os.path.join(out.base_dir, 'neutron_source'))
        outpath.mkdir(parents=True, exist_ok=True)



        print('Step 9.1: writing source term ...')
        source_sum= 0
        index= 0

        # z_core_flip= np.flip(geom.z_core) ## FLIPPING NOT NEEDED FOR VERA BECAUSE THE PLANES ARE ALREADY FROM BOTTOM TO TOP

        # compute total source strength
        for pin in self.sourcepinVERA:

            # sum source of neutrons
            source_sum += pin[7]

        with open(os.path.join(outpath, 'LWR-10-external_source_' + str(self.step) + 'VWS.ser'), 'w') as f:
            
            for pin in self.sourcepinVERA:

                index += 1

                # write source for pin
                print('Writing source for pin ...' + str(index))

                # compute pin coordinates (UPDATED)
                cA = (geom.nass + 1) / 2
                cP = (geom.npin + 1) / 2
                x_core = (pin[2] - cA) * geom.ass_pitch
                y_core = (cA - pin[1]) * geom.ass_pitch
                x_pin = x_core + (pin[5] - cP) * geom.pin_pitch
                y_pin = y_core + (cP - pin[4]) * geom.pin_pitch
                z_pin = (geom.z_core[pin[3]] + geom.z_core[pin[3]+1])/2
                node_height= geom.meshheight[pin[3]]
                
                # compute pin boundaries
                x_min= x_pin - geom.pin_pitch/2
                x_max= x_pin + geom.pin_pitch/2
                y_min= y_pin - geom.pin_pitch/2
                y_max= y_pin + geom.pin_pitch/2
                z_min= z_pin - node_height/2
                z_max= z_pin + node_height/2

                # fuel type
                fuel_type= pin[9]
                # write neutron source 
                if self.sourceoption == 'pointsource':
                
                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X):
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                if self.sourceoption == 'volumesource':

                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                    
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X): # NOTE: this is HARD CODED to work for the bottom quarter of the core!! This is why you have sx x_pin - x_max and not x_min - x_pin
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_pin:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")

                        elif ([pin[1], pin[2]] in self.trunc_ass_Y): # NOTE: this is HARD CODED to work for the bottom quarter of the core!! This is why you have sy y_min - y_pin and not y_pin -y_max 

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_pin:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")


                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                           
                f.write('1E-11 0.0\n') # insert first bin with zero value
                #insert energy spectrum specification
                print('Writing energy spectrum for pin ...' + str(index))
                for i in range(len(self.bins) - 1):
                    f.write(f"{self.bins[i+1]:.5e} {pin[8][i]:.5e}\n")
            
        with open(os.path.join(filepath), 'r') as f:
            lines = f.readlines()

        print ('Writing total source strength for normalization ...')
        

        with open(os.path.join(outpath, 'LWR-09-main_' + str(self.step) + 'VWS.ser'), 'w') as f:
            for i, line in enumerate(lines):
                if 'set srcrate' in line:
                    lines[i] = f'set srcrate {source_sum:.5e}\n'
                    break
            f.writelines(lines)

    def write_source_pinVERA2D(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs", filepath: Union[str, Path]) -> None:
        """
        Write 2D VERA pin-level neutron source terms to Serpent external source format.
        """
        # define output directory for source
        outpath = Path(os.path.join(out.base_dir, 'neutron_source'))
        outpath.mkdir(parents=True, exist_ok=True)



        print('Step 9.1: writing source term ...')
        source_sum= 0
        index= 0

        # NOTE: no flipping needed for VERA because the planes are already ordered from bottom to top

        # compute total source strength
        for pin in self.sourcepinVERA2D:

            # sum source of neutrons
            source_sum += pin[7]

        with open(os.path.join(outpath, 'LWR-10-external_source_' + str(self.step) + 'VWS.ser'), 'w') as f: # HARD CODED: HFP conditions
            
            for pin in self.sourcepinVERA2D:

                index += 1

                # write source for pin
                print('Writing source for pin ...' + str(index))

                # compute pin coordinates (UPDATED)
                cA = (geom.nass + 1) / 2
                cP = (geom.npin + 1) / 2
                x_core = (pin[2] - cA) * geom.ass_pitch
                y_core = (cA - pin[1]) * geom.ass_pitch
                x_pin = x_core + (pin[5] - cP) * geom.pin_pitch
                y_pin = y_core + (cP - pin[4]) * geom.pin_pitch
                z_pin = (geom.z_core[pin[3]] + geom.z_core[pin[3]+1])/2
                node_height= geom.meshheight[pin[3]]
                
                # compute pin boundaries
                x_min= x_pin - geom.pin_pitch/2
                x_max= x_pin + geom.pin_pitch/2
                y_min= y_pin - geom.pin_pitch/2
                y_max= y_pin + geom.pin_pitch/2
                z_min= z_pin - node_height/2
                z_max= z_pin + node_height/2

                # fuel type
                fuel_type= pin[9]
                # write neutron source 
                if self.sourceoption == 'pointsource':
                
                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X):
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        elif ([pin[1], pin[2]] in self.trunc_ass_Y):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sp {x_pin:.5e} {y_pin:.5e} {z_pin:.5e} sb {len(self.bins)} 1\n")

                if self.sourceoption == 'volumesource':

                    if (self.truncoption == False):
                        f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                    
                    elif (self.truncoption == True):

                        if ([pin[1], pin[2]] in self.trunc_ass_X):

                            if ([pin[4],pin[5]] in self.trunc_pinsym_X): # NOTE: this is HARD CODED to work for the bottom quarter of the core!! This is why you have sx x_pin - x_max and not x_min - x_pin
                                print(pin[4], pin[5])
                                print('symmetry axis')
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_pin:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_X) and ([pin[4],pin[5]] not in self.trunc_pin_X):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")

                        elif ([pin[1], pin[2]] in self.trunc_ass_Y): # NOTE: this is HARD CODED to work for the bottom quarter of the core!! This is why you have sy y_min - y_pin and not y_pin -y_max 

                            if ([pin[4],pin[5]] in self.trunc_pinsym_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_pin:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                            elif ([pin[4],pin[5]] not in self.trunc_pinsym_Y) and ([pin[4],pin[5]] not in self.trunc_pin_Y):
                                f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")

                        else:
                            f.write(f"\nsrc {int(index)} n sw {pin[7]/source_sum:.5e} sx {x_min:.5e} {x_max:.5e} sy {y_min:.5e} {y_max:.5e} sz {z_min:.5e} {z_max:.5e} sm {fuel_type} sb {len(self.bins)} 1\n")
                           
                f.write('1E-11 0.0\n') # insert first bin with zero value
                #insert energy spectrum specification
                print('Writing energy spectrum for pin ...' + str(index))
                for i in range(len(self.bins) - 1):
                    f.write(f"{self.bins[i+1]:.5e} {pin[8][i]:.5e}\n")
            
        with open(os.path.join(filepath), 'r') as f:
            lines = f.readlines()

        print ('Writing total source strength for normalization ...')
        

        with open(os.path.join(outpath, 'LWR-09-main_' + str(self.step) + 'VWS.ser'), 'w') as f: # HARD CODED: HFP conditions
            for i, line in enumerate(lines):
                if 'set srcrate' in line:
                    lines[i] = f'set srcrate {source_sum:.5e}\n'
                    break
            f.writelines(lines)

    # Source post-processing methods
    def timeaverage(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs", average_method: str = 'corexp') -> None:
        """
        Time-average the external source distribution over the cycle based on either equivalent
        full power days (efpd) or core exposure (corexp).

        This method collects the per-step external source and main files previously written by
        the write_source_ass / write_source_pin / write_source_pin2D methods, weights the source
        strengths and energy spectra of each state point by the corresponding time-integration
        weight, and writes a single cycle-averaged source specification and main input file.

        Parameters
        ----------
        cycle : Cycle
            Cycle object containing the state point information (cycleinfodays, cycleinfopow,
            cycleinfoexp), polaris options, energy group information and Polaris group widths.
        geom : Geometry
            Geometry object. Accepted for signature consistency with the other Source methods;
            not used by the averaging procedure.
        out : Outputs
            Outputs object for managing output directories. The per-step source files are read
            from the neutron_source directory and the averaged files are written to
            neutron_source/average.
        filepath : Union[str, Path]
            Path to the Serpent main input file used as template for the averaged main file.
        flag : str, optional
            Time-integration variable: 'efpd' weights by equivalent full power days, 'corexp'
            weights by core exposure increase. Default is 'corexp'.

        Returns
        -------
        None
            Writes LWR-10-external_source_avg.ser and LWR-09-main_avg.ser to the
            neutron_source/average directory.

        Notes
        -----
        - The current integration strategy neglects the last timestep. One alternative could be to neglect the first timestep instead or to use a trapezoidal integration scheme. 
          This is a modeling choice that can be revisited. See "A Versatile Methodology for Reactor Pressure Vessel Aging Assessments, R.Vuiart et al." for a detailed discussion.
        - Guide tubes and instrument tubes are filtered out via their zero source weight.
        - Averaged files are written to a dedicated sub-directory so that they are not picked
          up as state points on a subsequent call.

        See Also
        --------
        write_source_ass : Assembly-level source term output.
        write_source_pin : Pin-level source term output.
        """

        print('Selected class method to time average - Step 1: Averaging Source Distribution and Weight')

        days = cycle.cycleinfodays
        powlevel= cycle.cycleinfopow
        corexp= cycle.cycleinfoexp
        efpd_dt= np.zeros(len(days)-1)
        corexp_dt= np.zeros(len(days)-1)

        for i in range(len(days)-1):
            #efpd_dt[i]= (powlevel[i+1]/100)*(days[i+1]-days[i])    old
            efpd_dt[i]= 0.5*((powlevel[i+1]+powlevel[i])/100)*((days[i+1]-days[i])/days[-1])
            corexp_dt[i]= (corexp[i+1]-corexp[i])
            #efpd[i+1]= efpd[i]+ efpd_dt[i]

        # define weight to time average based on equivalent full power and exposure
        efpdw= efpd_dt/sum(efpd_dt)
        corexpw= corexp_dt/sum(corexp_dt)

        # define input/output directories for source averaging (consistent with write_source_* methods)
        srcpath = Path(os.path.join(out.base_dir, 'neutron_source'))
        avgpath = srcpath / 'average'
        avgpath.mkdir(parents=True, exist_ok=True)

        # collect the per-step files written by the write_source_* methods, keyed by step number
        steppattern = re.compile(r'_(\d+)(?:AWS|PWS|VWS)\.ser$')
        sourcefiles= {}
        mainfiles= {}
        for file in sorted(os.listdir(srcpath)):
            match = steppattern.search(file)
            if match is None:
                continue
            if file.startswith('LWR-10-external_source_'):
                sourcefiles[int(match.group(1))]= srcpath / file
            elif file.startswith('LWR-09-main_'):
                mainfiles[int(match.group(1))]= srcpath / file

        if not sourcefiles:
            print(f'No external source files found in {srcpath}: run the write_source_* methods first.')
            raise SystemExit

        #open sample file: the first available state point defines the source headings and positions
        with open(sourcefiles[min(sourcefiles)], 'r') as f:
            lines= f.readlines()

        #extract source heading
        headings= []
        posheadings= []
        for line in lines:
            if 'src' in line:
                data= line.split()
                #insert a filter to neglect guide tubes and instrument tubes
                if float(data[4]) != 0.0:
                    # save source name
                    head= str('src ') + str(data[1])
                    headings.append(head)
                    # save source position
                    pos= data[5:]
                    posheadings.append(pos)

        # Create a dictionary to store the source data for each heading
        if (cycle.polarisoption == 0) or (cycle.polarisoption == 1) :
            source_data = {heading: {'posheading': posheading, 'weight': 0, 'energy': np.zeros(len(self.bins)), 'ubound': np.zeros(len(self.bins))} for heading, posheading in zip(headings, posheadings)}
            rangeselection= len(self.bins)
        elif (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
            source_data = {heading: {'posheading': posheading, 'weight': 0, 'energy': np.zeros(cycle.groups), 'ubound': np.zeros(cycle.groups)} for heading, posheading in zip(headings, posheadings)}
            rangeselection= cycle.groups

        # Iterate over all files once and accumulate data
        for file_number, file in sorted(sourcefiles.items()):
            print(f'Processing file: {file.name}')
            if file_number != (len(cycle.cycleinfopow)-1): # the current integration strategy neglects the last timestep

                with open(file, 'r') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    if 'src' in line:
                        data= line.split()
                        heading= str('src ') + str(data[1])
                        if heading in source_data:

                            if average_method == 'efpd':
                                source_data[heading]['weight'] += float(data[4])*efpdw[file_number]
                            elif average_method == 'corexp':
                                source_data[heading]['weight'] += float(data[4])*corexpw[file_number]


                            for j in range(rangeselection):
                                energy_line = lines[i + 1 + j]
                                energy_data = energy_line.split()

                                if average_method == 'efpd':
                                    source_data[heading]['energy'][j] += float(energy_data[1])*efpdw[file_number]
                                elif average_method == 'corexp':
                                    source_data[heading]['energy'][j] += float(energy_data[1])*corexpw[file_number]

                                # the group structure is a property of the energy grid, not of the
                                # time weighting: it is read for every flag (used by polarisoption 2/3)
                                source_data[heading]['ubound'][j] = float(energy_data[0])

        # Normalize weights
        print('Normalizing weights')
        total_weight = sum(data['weight'] for data in source_data.values())
        for heading in source_data:
            source_data[heading]['weight'] /= total_weight

        # Integrate and normalize spectra
        print('Integrating and normalizing energy spectra')
        bin_widths = np.diff(self.bins)
        polaris_widths= cycle.polariswidth
        #print(polaris_widths)
        for heading in source_data:
            if (cycle.polarisoption == 0) or (cycle.polarisoption == 1):
                integral = np.sum(source_data[heading]['energy'][1:] * bin_widths)
                source_data[heading]['energy'][1:] /= integral
            elif (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                integral = np.sum(source_data[heading]['energy'][1:] * polaris_widths)
                source_data[heading]['energy'][1:] /= integral

        # Write time-averaged source
        with open(os.path.join(avgpath,'LWR-10-external_source_avg.ser'), 'w') as f:
            for heading in source_data:
                new_heading = heading + f" n sw {source_data[heading]['weight']:.5e} " + " ".join(source_data[heading]['posheading'])
                f.write(new_heading + "\n")
                f.write("1E-11 0.0 \n")
                for j in range(rangeselection - 1):
                    if (cycle.polarisoption == 0) or (cycle.polarisoption == 1):
                        f.write(f"{self.bins[j+1]:.5e} {source_data[heading]['energy'][j+1]:.5e}\n")
                    elif (cycle.polarisoption == 2) or (cycle.polarisoption == 3):
                        f.write(f"{source_data[heading]['ubound'][j+1]:.5e} {source_data[heading]['energy'][j+1]:.5e}\n")

        print('Selected class method to time average - Step 2: Averaging Source Distribution and Weight')

        srcrate= np.zeros(len(mainfiles))

        for file_number, file in sorted(mainfiles.items()):
            print(f'Processing file for source rate: {file_number}')

            with open(file, 'r') as f:
                lines= f.readlines()
            for line in lines:
                if 'set srcrate' in line:
                    data= line.split()
                    srcrate[file_number]= float(data[2])
        # define weight to time average based on equivalent full power
        if average_method == 'efpd':
            wsrcrate= srcrate[:-1]*efpdw # the current time integration strategy neglects the last timestep
            avg_srcrate= sum(wsrcrate)/sum(efpdw)
        elif average_method == 'corexp':
            wsrcrate= srcrate[:-1]*corexpw # the current time integration strategy neglects the last timestep
            avg_srcrate= sum(wsrcrate)/sum(corexpw)

        # read the template main file used to write the averaged source rate
        with open(sourcefiles[min(sourcefiles)], 'r') as f:
            lines= f.readlines()

        # create main file based on average srcrate
        with open(os.path.join(avgpath,'LWR-09-main_avg.ser'), 'w') as f:
            for line in lines:
                if 'set srcrate ' in line:
                    sub= 'set srcrate ' + f"{avg_srcrate:.5e}\n"
                    f.write(sub)
                else:
                    f.write(line)