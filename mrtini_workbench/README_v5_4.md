# Workbench v5.4 — fast acquisitions and magnetic timeline

## Run mode

The UI defaults to **Fast · image + ADC**. Choose **With diagnostics** in the acquisition inspector or the timed designer before running to enable the full analysis workflow. The selection does not itself trigger a simulation and is shared by both views. The legacy API default remains `diagnostics=True` for compatibility; pass `diagnostics=False` in a job's protocol to request fast mode.

Fast runs still build/reuse the physical phantom, bake/reuse RF coefficients, construct/reuse the PDG, execute the full simulator, reconstruct the image and save the complex ADC, trajectory and reconstruction. They omit phantom-map rasterization, graph overview, compressed node snapshot and automatic voxel analysis. The physical sequence strip and encoding trajectory are still available. The Analyze views explain how to rerun with diagnostics. Diagnostic replays remain on demand in diagnostic mode. Detailed full-edge archiving remains an explicit operation. Run mode is excluded from native preparation/cache keys. Saving raw acquisition results and browser display still have overhead; timings include separate export and display stages. Compare warm runs with the same protocol and inputs for meaningful latency measurements.

## Magnetic timeline

Magnetic snap is enabled by default. Drag or resize an event near any start/end boundary: an eight-screen-pixel proximity threshold follows the zoom level. Same-channel boundaries take precedence in ties. A vertical guide shows the snapped time. Hold Alt or turn Magnetic snap off for ordinary fine time-grid edits. ADC duration is quantized to whole samples; RF duration is quantized to the supported 10 us raster. If a resize cannot exactly hit a boundary, no snap is indicated. Overlaps across channels remain permitted; overlapping same-channel events remain rejected by the existing native compiler. This is not ripple editing and does not move neighbouring events.

## Encoding protection

New timed FLASH and spin-echo templates explicitly mark the generated Gx prephaser, Gy phase encoding, Gx readout, ADC, Gx rewinder and Gy rewinder. These six events are protected by default (dashed outline, no resize handle, read-only inspector). FOV, acquisition matrix and readout dwell regenerate the tagged Cartesian encoding around the same nominal readout centre. If shots previously matched the matrix they follow a matrix change; manually chosen shot counts remain unchanged. The native compiler continues to validate timing. The separate phantom and reconstruction matrix controls are unaffected.

Enable **Edit encoding gradients** to drag, resize or edit the protected events. Manual edits are retained when relocking. Geometry controls are disabled during manual encoding editing to avoid overwriting changes. **Regenerate encoding** explicitly restores the tagged group's nominal Cartesian moments/timings from the geometry inputs. It requires a complete group; if members have been deleted, restore a template or use manual timing. The acquisition sidebar no longer edits a custom design's FOV/matrix independently of its recipe. RF, slice-selection gradients, crushers, spoilers and other custom events are not automatically protected. Custom gradients can be marked protected in the inspector. Old saved timed recipes without role metadata are not guessed from their shapes: reselect a starting recipe or mark gradients manually. The high-level recipe editor continues to synthesize its readout gradients from readout parameters.

## Tests

`python -m pytest -q` from this folder. `node test_timeline_tools.cjs` for pure timeline helper tests. Browser smoke tests used the offline bridge to the real FastAPI preview and timed design endpoints. With the recovered native CPU fixture available: 79 tests passed, 2 skipped (optional NUFFT and DICOM libraries). Native CUDA timing and numerical validation require the user's installation. The production simulator source is unchanged.
