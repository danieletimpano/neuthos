# examples/run_parcs_cycle01.py
import neuthos as nt
from datetime import datetime
import numpy as np

from neuthos import source

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = nt.Outputs(base_dir=f"output_{timestamp}")

# Define the fullcore map
fullcore= [                                       [2,8],  [2,9],  [2,10], 
                                  [3,6],  [3,7],  [3,8],  [3,9],  [3,10],  [3,11],  [3,12],
                          [4,5],  [4,6],  [4,7],  [4,8],  [4,9],  [4,10],  [4,11],  [4,12],  [4,13],
                  [5,4],  [5,5],  [5,6],  [5,7],  [5,8],  [5,9],  [5,10],  [5,11],  [5,12],  [5,13],  [5,14],
           [6,3], [6,4],  [6,5],  [6,6],  [6,7],  [6,8],  [6,9],  [6,10],  [6,11],  [6,12],  [6,13],  [6,14],  [6,15],
           [7,3], [7,4],  [7,5],  [7,6],  [7,7],  [7,8],  [7,9],  [7,10],  [7,11],  [7,12],  [7,13],  [7,14],  [7,15],
    [8,2], [8,3], [8,4],  [8,5],  [8,6],  [8,7],  [8,8],  [8,9],  [8,10],  [8,11],  [8,12],  [8,13],  [8,14],  [8,15],  [8,16],
    [9,2], [9,3], [9,4],  [9,5],  [9,6],  [9,7],  [9,8],  [9,9],  [9,10],  [9,11],  [9,12],  [9,13],  [9,14],  [9,15],  [9,16],
    [10,2],[10,3],[10,4], [10,5], [10,6], [10,7], [10,8], [10,9], [10,10], [10,11], [10,12], [10,13], [10,14], [10,15], [10,16],
           [11,3],[11,4], [11,5], [11,6], [11,7], [11,8], [11,9], [11,10], [11,11], [11,12], [11,13], [11,14], [11,15],
           [12,3],[12,4], [12,5], [12,6], [12,7], [12,8], [12,9], [12,10], [12,11], [12,12], [12,13], [12,14], [12,15], 
                  [13,4], [13,5], [13,6], [13,7], [13,8], [13,9], [13,10], [13,11], [13,12], [13,13], [13,14],
                          [14,5], [14,6], [14,7], [14,8], [14,9], [14,10], [14,11], [14,12], [14,13],
                                  [15,6], [15,7], [15,8], [15,9], [15,10], [15,11], [15,12],
                                                  [16,8], [16,9], [16,10]]


# Define the source input
inputsource= [  
                                                 [9,14],  [9,15],  [9,16],
                                                 [10,14], [10,15], [10,16],
                                         [11,13],[11,14], [11,15],
                                 [12,12],[12,13],[12,14], [12,15], 
                         [13,11],[13,12],[13,13],[13,14],
                 [14,10],[14,11],[14,12],[14,13],
         [15,9], [15,10],[15,11],[15,12],
         [16,9], [16,10]   ]

# Define reactor geometry
geom = nt.Geometry(
    naxial=32,
    nass=17,
    npin=15,
    latsym= "se",
    ngtubes= 20, 
    nitubes= 1,
    ass_pitch=21.60912,
    pin_pitch=1.43511,
    pin_radius= 0.4677,
    active_height= 368.26,
    coremap= fullcore,
    source= inputsource,
    ngtubesqtr= 6,
    nitubesqtr= 1
)

# Compute source coordinates from PARCS
geom.compute_radial_coordinates_PARCS()
geom.read_axial_mesh_from_parcs("parcs_files/TP_cyc1_HFP3D.parcs_out")
out.write_PARCS_pin_coordinates(geom, "00.parcs_pin_coordinates.csv")

# Extract basic cycle information
cycle = nt.Cycle(
    nassembly_with_reflectors= 221,
    polarisoption= 0,                    # 0 uses literature assumptions for nubar/sigma_f/Er
    nsteps= 6,
    asspower= 14.0127e6
)
cycle.extract_cycle_info("parcs_files/TP_cyc1_HFP3D.parcs_dpl")
cycle.extract_coolant_info(geom, "parcs_files/TP_cyc1_HFP3D.parcs_cyc-01")
out.write_custom_csv(cycle.coolant_density[0], "01.coolant_density_0.csv")

# Extract assembly axial and radial configuration
cycle.extract_assyaxial("parcs_files/depl_assm.geom")
out.write_assyaxial(cycle, "02.assembly_axial_configuration.csv")
cycle.extract_assyradial(geom, "parcs_files/c1_dep_30nodes.inp")
out.write_assyradial(cycle, "03.assembly_radial_configuration.csv")

# Extract lattice composition
cycle.extract_lattype("parcs_files/pmax.dir")
out.write_lattype_to_csv(cycle, "04.lattice_type_data.csv")
cycle.extract_latcomp_scale63(geom,"parcs_files/library")
out.write_latcomp_to_csv(cycle, "05.lattice_composition_data.csv")
cycle.extract_exposure(geom, "parcs_files/TP_cyc1_HFP3D.parcs_cyc-01")
out.write_exposureindex_to_csv(cycle, "06.cycle_exposure.csv")
out.write_custom_csv(cycle.exposure[:, :, 5], "07.exposure_step0.csv")  # Save exposure at step 0 as example

# Extract assembly power data
cycle.extract_asspower(geom, out, "parcs_files")
out.write_asspower_to_csv(cycle, "08.assembly_power_data.csv")

# Extract pin power data
cycle.extract_pinpower(geom, out, "parcs_files")
out.write_pinpower_to_csv(cycle, "09.pin_power_data.csv")

# Build 3D assembly source
assembly_source = nt.Source(
    step = 0,
    bins = np.linspace(0, 20, 200),
    truncoption= True,
    trunc_ass_X= [[2,9],[3,9]],
    trunc_ass_Y= [[9,2],[9,3],[9,4]],
    assytype_to_mcmaterial= {"10": "fuel_1",
                             "20": "fuel_2",
                             "22": "fuel_2",
                             "30": "fuel_3",
                             "32": "fuel_3",
                             "40": "fuel_4",
                             "50": "fuel_5",
                             "60": "fuel_6"}
)
assembly_source.build_source_ass(cycle, geom, out)
out.write_asssourceinfo_to_csv(assembly_source, "09.assembly_source_info.csv")
out.visualize_source_ass(geom, assembly_source, '3D')
out.visualize_source_ass(geom, assembly_source, '2D')
assembly_source.write_source_ass(
    cycle, geom, out, "serpent_files/LWR-09-main_avg.ser" # provide path to sample input to write set srcrate
)

# Build 3d pin source
pin_source = nt.Source(
    step= 0,
    bins= np.linspace(0, 20, 200),
    truncoption= True,
    sourceoption= 'pointsource',
    trunc_ass_X= [[2,9], [3,9]],
    trunc_ass_Y= [[9,2],[9,3],[9,4]],
    trunc_pin_X= [[x,y] for x in np.linspace(1,15,15).astype(int) for y in np.linspace(9,15,7).astype(int)],
    trunc_pin_Y= [[x,y] for x in np.linspace(9,15,7).astype(int) for y in np.linspace(1,15,15).astype(int)],
    assytype_to_mcmaterial= {"10": "fuel_1",
                             "20": "fuel_2",
                             "22": "fuel_2",
                             "30": "fuel_3",
                             "32": "fuel_3",
                             "40": "fuel_4",
                             "50": "fuel_5",
                             "60": "fuel_6"}
)

pin_source.build_source_pin(cycle, geom, out)
out.write_pinsourceinfo_to_csv(pin_source, "09.pin_source_info.csv")
out.visualize_source_pin(geom, pin_source, '2D')
pin_source.write_source_pin(
    cycle, geom, out, "serpent_files/LWR-09-main_avg.ser" # provide path to sample input to write set srcrate
)