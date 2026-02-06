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

@dataclass
class Source  :
    # --- source inputs ---
    step: int                                                                 # step used for source term extraction
    bins: np.ndarray                                                          # linspace used to define energy bins
    truncoption: bool                                                         # option to control whether to truncate the fission spectrum at the upper energy bound of the bins (0: no truncation, 1: truncation)
    assytype_to_mcmaterial: dict                                              # dictionary to store the link between serpent material and parcs assemblies
    sourceoption: str = 'pointsource'                                         # option to control whether to write point source or volumetric source
    trunc_ass_X: List[Tuple[int, int]] = None                                 # assemblies along X truncation line
    trunc_ass_Y: List[Tuple[int, int]] = None                                 # assemblies along Y truncation line
    trunc_pin_X: List[Tuple[int, int]] = None                                 # pins along X truncation line
    trunc_pin_Y: List[Tuple[int, int]] = None                                 # pins along Y truncation line

    # --- computed / parsed outputs ---
    sourceassy : List[Tuple[int, int, int]] = field(default_factory=list)                       # list of tuples with assembly coordinates and type (i,j,assytype)
    sourcepin : List[Tuple[int, int, int, int, int, int, float]] = field(default_factory=list)  # list of tuples with pin coordinates and type (step, i_index, j_index, k_index, x_index, y_index, power_value)

    def build_source_ass(
        self, cycle: "Cycle", geom: "Geometry", out: "Outputs") -> None:

        print('Preparing data for source term ...')
        
        # for step in steps:

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(41,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[], 'eubound':[]}
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
        plt.savefig(plotpath / f'assembly_burnup_comparison.png')

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
        plt.savefig(plotpath / f'3D_assembly_burnup_comparison.png', bbox_inches='tight')

        # save the data for these assemblies to a csv file
        with open(checkspath / f'burnup_profile.csv', 'w', newline='') as csvfile:
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
        plt.savefig(plotpath / f'assembly_u235_comparison.png')

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
        plt.savefig(plotpath / f'assembly_u238_comparison.png')

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
        plt.savefig(plotpath / f'assembly_pu239_comparison.png')

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
        plt.savefig(plotpath / f'assembly_pu241_comparison.png')

        # save plutonium content to txt
        with open(checkspath / f'plutonium_content.txt', 'a', newline='') as f:
            
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
        plt.savefig(plotpath / f'fission_spectrum.png')

        # Save fission spectrum to CSV
        if (cycle.polarisoption==0) or (cycle.polarisoption==1) :
            with open(checkspath / f'fission_spectrum.csv', 'w', newline='') as csvfile:
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
                x_core = (ass[2] - (geom.nass + 1) // 2) * geom.ass_pitch
                y_core = ((geom.nass + 1) // 2 - ass[1]) * geom.ass_pitch
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

        print('Step 8.1: preparing data for source term ...')

        # list of dictionaries to store the assembly information
        assembly_list= []

        # create a dictionary to store assembly information based on the assembly location
        for assy in cycle.assyradial:
            assembly= {'coordinates': [assy[0],assy[1]], 'assytype':assy[2], 'r-f-d':[], 'burnupID':[], 'nz':np.linspace(41,1,1), 'latID':[], 'buprofile':[], 'bu_clos':[], 'nubar':[], 'ERC':[], 'U235':[], 'U238':[], 'Pu239':[], 'Pu241':[], 'nu': [], 'erf':[], 'chi':[], 'F':[]}
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
                assembly['latID']= [0]*41
                assembly['burnupID']= 0
                assembly['buprofile']= [0]*41
                assembly['bu_clos']= [0]*41
                assembly['nubar']= 0
                assembly['ERC']= 0
                assembly['U235']= [0]*41
                assembly['U238']= [0]*41
                assembly['Pu239']= [0]*41
                assembly['Pu241']= [0]*41
        
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
        plt.savefig(plotpath / 'assembly_burnup_comparison.png')

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
        plt.savefig(plotpath / '3D_assembly_burnup_comparison.png', bbox_inches='tight')

        # save the data for these assemblies to a csv file
        with open(checkspath / 'burnup_profile.csv', 'w', newline='') as csvfile:
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
        plt.savefig(os.path.join(plotpath, 'fission_spectrum.png'))

        # Save fission spectrum to CSV
        with open(os.path.join(checkspath, 'fission_spectrum.csv'), 'w', newline='') as csvfile:
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


        # save total source rate to check file
        with open(os.path.join(checkspath, 'source_rate_pin3D.txt'), 'w', newline='') as f:
            f.write('Total source rate: ' + str(np.sum([x[7] for x in self.sourcepin])))

    def write_source_pin(
            self, cycle: "Cycle", geom: "Geometry", out: "Outputs", filepath: Union[str, Path]) -> None:


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

                # compute pin coordinates
                x_core = (pin[2] - (geom.nass + 1) // 2) * geom.ass_pitch
                y_core = ((geom.nass + 1) // 2 - pin[1]) * geom.ass_pitch
                x_pin = x_core + (pin[5] - (geom.npin + 1) // 2) * geom.pin_pitch
                y_pin = y_core + ((geom.npin + 1) // 2 - pin[4]) * geom.pin_pitch
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

