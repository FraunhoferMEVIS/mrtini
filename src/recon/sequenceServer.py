"""!
@details Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
         AGPLv3-clause License

         The software is not qualified for use as a medical product or as part
         thereof. No bugs or restrictions are known.
"""

import cProfile
import cProfile
import pstats
import sys
from typing import Any


sys.path.append('./')
from src.mrtini.config import PAD, SLICE_THICKNESS
from src.mrtini.pre_pass_rust_call import compute_graph_rust
from src.voxelgridphantom.voxel_grid_phantom import VoxelGridPhantom


import json
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os
import base64
from io import BytesIO
import random
import ismrmrd
import shutil
import zipfile
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.sequence.sequence import Sequence
import mrpy_ismrmrd_tools as ismrmrd_tools
import mrpy_io_tools as io_tools
import torch
from src.mrtini.mrtini_main import MRtiniSim
from src.mrtini.execute_graph import execute_graph


def get_gammastar_recon_endpoint():
    host = os.getenv("GAMMASTAR_RECON_HOST", "localhost")
    port = int(os.getenv("GAMMASTAR_RECON_PORT", "9002"))
    return host, port

def build_prepass_inputs_from_hybrid(seq, art):
    # 1) map hybrid bloch spans -> per-repetition key
    bloch_span = [si for si,s in enumerate(seq.mrtini_hybrid) if s["kind"]=="bloch"]
    R = len(list(seq))  # repetitions you pass to Rust
    assert len(bloch_span) >= R, "Not enough bloch spans for repetitions"
    rep_keys = [art.span_key[bloch_span[r]] for r in range(R)]

    # 2) assign pulse_id for each unique key (stable ordering)
    uniq_keys = list(dict.fromkeys(rep_keys))  # preserve first-seen order
    key_to_pid = {k:i for i,k in enumerate(uniq_keys)}
    pulse_id_per_rep = torch.tensor([key_to_pid[k] for k in rep_keys], dtype=torch.int32)

    # 3) build taps/support banks aligned with pid
    taps_bank = [art.pulses[k].combined_taps for k in uniq_keys]
    supp_bank = [art.pulses[k].water.support      for k in uniq_keys]

    return rep_keys, uniq_keys, key_to_pid, pulse_id_per_rep, taps_bank, supp_bank,


def print_info_server():
    print("########################################################\n\n"
          "gammaSTAR Server v1.0.1 Release\n")

    print("The software is not qualified for use as a medical product or as part\n"
          "thereof. No bugs or restrictions are known. Delivered ‘as is’ without\n"
          "specific verification or validation.\n"
          "(C) Fraunhofer MEVIS 2025\n"
          "Contact: Daniel Christopher Hoinkiss (daniel.hoinkiss@mevis.fraunhofer.de)\n")

    print("List of included third-party software\n"
          "-\n"
          "numpy (BSD License)\n"
          "matplotlib (Matplotlib License)\n"
          "ismrmrd (ISMRMRD SOFTWARE LICENSE JULY 2013)\n"
          "h5py (BSD License)\n"
          "pydicom (MIT License)\n"
          "fastapi (MIT License)\n"
          "MRzero (AGPL-3.0 License)\n"
          "-\n"
          "Detailed information about third-party licence conditions can\n"
          "be found in the 'third_party_licenses' folder.\n\n"
          "########################################################\n")

def print_info_parser():

    print("########################################################\n\n"
          "gammaSTAR Parser v1.0.1 Release\n")

    print("The software is not qualified for use as a medical product or as part\n"
          "thereof. No bugs or restrictions are known. Delivered ‘as is’ without\n"
          "specific verification or validation.\n"
          "(C) Fraunhofer MEVIS 2025\n"
          "Contact: Snawar Hussain (snawar.hussain@mevis.fraunhofer.de)\n")

    print("List of included third-party software\n"
          "-\n"
          "torch (PyTorch) – BSD License\n"
          "numpy – BSD License\n"
          "matplotlib – Matplotlib License (based on PSF)\n"
          "MRzero - AGPL-3.0 license\n"
          "serde (MIT or Apache-2.0)\n"
          "serde_json (MIT or Apache-2.0)\n"
          "thiserror (MIT or Apache-2.0)\n"
          "ezpc (MIT)\n"
          "interp (MIT or Apache-2.0)\n"
          "serde_with (MIT or Apache-2.0)\n"
          "pyo3 (MIT or Apache-2.0)\n"
          "-\n"
          "Detailed information about third-party licence conditions can\n"
          "be found in the 'third_party_licenses' folder.\n\n"
          "########################################################\n")

time.sleep(0.1)
print_info_server()
time.sleep(0.1)
print_info_parser()
time.sleep(0.1)
Nx = Ny = 128
# phantom generation
phantom2d = VoxelGridPhantom.load('data/subject04_3T-highres-fat_custom_dense.npz')
# phantom2d = phantom2d.interpolate(Nx, Ny, 64).slices(list(np.linspace(0, 63, 64)))  # select slices
# phantom2d = phantom2d.interpolate(Nx, Ny, 32).slices([16])  # select slices
phantom2d = phantom2d.interpolate(Nx, Ny, 30).slices([14])  # select slices
# phantom2d = phantom2d.interpolate(Nx, Ny, 512).slices(list(map(int, np.arange(236,272,1))))  # select slices
FOV = [0.256, 0.256, SLICE_THICKNESS]  # in meters

# phantom2d.size = torch.tensor([0.256, 0.256, SLICE_THICKNESS])  # we need correct size for slice profile calculation
phantom2d.size[-1] = torch.tensor([FOV[-1]])  # we need correct size for slice profile calculation
# phantom2d.plot_mask()

phantom2d = phantom2d.build(voxel_shape="sinc")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start webserver
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

httpd = FastAPI(lifespan=lifespan)

# URL for the webserver
origins = ['http://localhost:4200',
           'http://localhost',
           'https://gammastar.mevis.fraunhofer.de']

# Add CORS middleware to allow cross-origin requests
httpd.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Access-Control-Allow-Origin", "Content-Type"],
    expose_headers=["Access-Control-Allow-Origin"],
    max_age=5
)

# Ensure /data_export is always an empty directory
if os.path.exists("/data_export"):
    shutil.rmtree("/data_export")
os.makedirs("/data_export", exist_ok=True)
httpd.mount("/data_export/", StaticFiles(directory="/data_export"), 
            name="data_export")

@httpd.get('/')
def default():
    return {'message': 'Hello World'}

# Simulate sequence for simulation
def reconstruct_image(sequenceData: dict[str, Any], signal: list[float], imgBufferTrajectory: BytesIO):
    output = []
    raw_adc_representations = []
    for rawRepr in sequenceData['raw_reps']:
        output.append(json.dumps(rawRepr))
        if rawRepr['has_adc']:
            raw_adc_representations.append(rawRepr)

# gammaSTAR recon
    np_arr = np.asarray(signal.cpu().numpy())
    sequenceData['protocol']['read_oversampling'] = (sequenceData['protocol']['read_oversampling'] + 1)
    sequenceData['protocol']['slice_oversampling'] = sequenceData['protocol']['slice_oversampling'] + 1
    sequenceData['protocol']['phase_oversampling'] = sequenceData['protocol']['phase_oversampling'] + 1

    read_size = sequenceData['root']['acq_size']['1'] * (sequenceData['protocol']['read_oversampling']) * sequenceData['protocol']['read_partial_fourier']
    np_reshape = np_arr.reshape(-1, 1, sequenceData['expo'].get('adc_samples', read_size)).T

    adc_points = sequenceData['trajectory']['adc_points']
    list_of_acqs = ismrmrd_tools.numpy_and_raw_rep_to_acq(np_reshape, raw_adc_representations, adc_points)
    xml_header = ismrmrd_tools.gstar_to_ismrmrd_hdr(sequenceData['protocol'], sequenceData['info'], sequenceData['expo'], sequenceData['sys'], sequenceData['root'], opt_seq_json=sequenceData['sequence'])

    response = {"dicom": "", 
    			"trajectory": [], 
    			"kspace": [], 
    			"recon": [], 
    			"raw": "", 
    			"recon_history": ""}

    os.makedirs('/data_export', exist_ok=True)
    meas_id = str(random.randint(10000, 99999))
    raw_filename = os.path.join('/data_export/' + sequenceData['name'].replace(' ', '_') + meas_id + '_raw.h5')

    try:
        dset = ismrmrd.Dataset(raw_filename, 'dataset', create_if_needed=True)
        dset.write_xml_header(xml_header)

        for i, acq in enumerate(list_of_acqs):
            dset.append_acquisition(acq)
        response['raw'] = raw_filename
        print(sequenceData['name'] + meas_id + '_raw.h5 written')
    finally:
        try:
            dset.close()
        except Exception:
            pass

    if 'skipReconstruction' not in sequenceData or sequenceData['skipReconstruction'] != True:
        recon_host, recon_port = get_gammastar_recon_endpoint()
        con = ismrmrd_tools.gstar_recon_emitter(recon_host, recon_port, list_of_acqs, xml_header, 1.0, "")
        if con != False:
            ismrmrd_tools.gstar_recon_injector(con)
            con.shutdown_close()
            recon_images = []

            if len(con.images) > 0:

                for img_idx in range(len(con.images)):
                    image_data = con.images[img_idx].data[0, :, :, :]
                    for slice_index in range(len(image_data)):
                        image_slice = image_data[slice_index]

                        # Normalize image
                        image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice))

                        # save to h5
                        recon_images.append(image_slice.astype(np.float32))

                        # Encode as PNG
                        imgBufferRecon = BytesIO()
                        plt.figure(figsize=(20, 20))
                        plt.imshow(image_slice, cmap='gray')
                        plt.axis('off')
                        plt.gca().set_axis_off()
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        plt.savefig(imgBufferRecon, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
                        imgBufferRecon.seek(0)
                        plt.close()
                        b64Recon = base64.b64encode(imgBufferRecon.getvalue())
                        response['recon'].append(b64Recon.decode())

                response['recon_history'] = con.images[0].meta['ImageHistory'][2]

                logger.info(f"Sent reconstruction history: {con.images[0].meta['ImageHistory'][2]}")

                dicom_dir = './data_export/dicom_' + sequenceData['name'].replace(' ', '_') + '_' + meas_id
                os.makedirs(dicom_dir, exist_ok=True, )
                zip_filename = os.path.join(dicom_dir, f"{sequenceData['name'].replace(' ', '_')}_{meas_id}_dicom.zip")
                for recv_image in con.images:
                    io_tools.write_dcm_from_ismrmrd_image(recv_image, dicom_dir)

                dicom_files = os.listdir(dicom_dir)

                with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    for dicom in dicom_files:
                        dicom_filename = os.path.join(dicom_dir, dicom)
                        zipf.write(dicom_filename, arcname=dicom)

                response['dicom'] = zip_filename

    if imgBufferTrajectory:
        b64Trajectory = base64.b64encode(imgBufferTrajectory.getvalue())
        response['trajectory'].append(b64Trajectory.decode())

    return JSONResponse(content=response)

def simulate_mr0(sequenceData, state_count=200, min_state_mag=1e-4,
                 tap_threshold=1e-5, plot_sp=False):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    global phantom2d
    output = []
    raw_adc_representations = []
    for rawRepr in sequenceData['raw_reps']:
        output.append(json.dumps(rawRepr))
        if rawRepr['has_adc'] == True:
            raw_adc_representations.append(rawRepr)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, 
                                     encoding='utf-8', suffix='.seq') as tempFile:
        tempFile.write('\n'.join(output))
        sequenceFile = tempFile.name

    seq = Sequence.import_file_for_mrtini_sim(sequenceFile)
    art = MRtiniSim(device="cuda", seq=seq).bake_bloch_kernels(
        phantom_data=phantom2d,
        z_span_m=FOV[-1],
        global_RF=False, ##  Change to True to use global RF ~ hard pulse approximation for Faster sim
    )
    
    (rep_keys, uniq_keys,
     key_to_pid, pulse_id_per_rep, 
     taps_bank, supp_bank) = build_prepass_inputs_from_hybrid(seq, art)
    
    graph = compute_graph_rust(
        seq, phantom2d,
        pulse_id_per_rep,
        taps_bank,
        supp_bank,
        art.z_m,
        slice_thickness=SLICE_THICKNESS, pad=PAD,
        max_state_count=state_count, min_state_mag=min_state_mag
    )
    ## uncomment to plot first two reps of the graph for debugging
    # render_pdg_pyvis(graph[:3])
    
    seq = seq.cuda()
    phantom2d = phantom2d.cuda()

    signal = execute_graph(graph,
                           seq,
                           phantom2d,
                           rep_keys,
                            art,
                           SLICE_THICKNESS=SLICE_THICKNESS,

                          )


    
    # kspace plot (code adapted from pypulseq sequence.plt_kspace_trajectory)
    plotting_dims = 'xy'
    figsize = (20, 20)
    dim_map = {'x': 0, 'y': 1, 'z': 2}
    seq = seq.cpu()
    kspace = seq.get_full_kspace()
    adc_mask = [rep.adc_usage > 0 for rep in seq]
    matplotlib.rcParams.update({'font.size': 35})
    cmap = plt.get_cmap('rainbow')
    plt.figure(figsize=figsize)

    for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
        kx = rep_traj[:, dim_map[plotting_dims[0]]]
        ky = rep_traj[:, dim_map[plotting_dims[1]]]

        plt.plot(kx, ky, c=cmap(i / len(kspace)))
        plt.plot(kx[mask], ky[mask], 'r.')
        plt.plot(kx[~mask], ky[~mask], 'k.')
    plt.xlabel(f"$k_{plotting_dims[0]}$")
    plt.ylabel(f"$k_{plotting_dims[1]}$")
    plt.grid()

    imgBufferTrajectory = BytesIO()
    plt.savefig(imgBufferTrajectory, format='png')
    imgBufferTrajectory.seek(0)

    if os.path.exists(sequenceFile):
        os.remove(sequenceFile)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return signal, imgBufferTrajectory

# # Endpoint for running simulation
# @httpd.post('/run_sim')
def run_sim(sequenceData_path:str):
    logging.info('Run simulation sequence')
    with open(sequenceData_path, 'r') as f:
        sequenceData = json.load(f)
        f.close()
    signal, imgBufferTrajectory = simulate_mr0(sequenceData)
    return reconstruct_image(sequenceData, signal, imgBufferTrajectory)
def main():
    sequenceData_path = (
        f"src/sequence/gstar_seqs/EPI_sequence_sequenceData.json"
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    response = run_sim(sequenceData_path=sequenceData_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(response)
    return response


if __name__ == "__main__":
    profiler = cProfile.Profile()

    profiler.enable()
    main()
    profiler.disable()

    profiler.dump_stats("prof_run_sim.out")

    p = pstats.Stats("prof_run_sim.out")
    p.strip_dirs().sort_stats("cumtime").print_stats(40)
