# contour_integration_pytorch

## Models
- Proposed models are defined in ./models/new_piech_models.py
- Control models are defined in ./models/new_control_models.py

## Data
- To generate training data for the synthetic contour fragments task: ./generate_data/generate_contour_data_set.py
- To generate data for the contour tracing in natural images task: generate_pathfinder_dataset.py
  - Requires the BIPED dataset (https://github.com/xavysp/MBIPED).

## Train
- To train models on the synthetic contour fragments tasks: train_contour_data_set.py
  - Requires the synthetic fragments dataset. See Data Section
- To trains models on the edge detection in natural images task: train_biped_data_set_3.py
   - Requires the BIPED dataset (https://github.com/xavysp/MBIPED).
- To train models on the contour tracing in natural images task:
  - Requires the contour tracing in natural images dataset. See Data Section.
  
## Analysis:
- For analyzing the effects of contour length on contour gain on models trained on sythetic contour fragments task: experiment_gain_vs_contour_length.py
- For analyzing the effects of fragment spacing on contour gain on models trained on sythetic contour fragments task: experiment_gain_vs_spacing.py
- The script: run_both_experiments.py will run the analysis of both contour length and fragment spacing on contour gain for a trained model.
- For analyzing the effects of fragment spacing on contour gain on models trained on stracing contours in natural images: experiment_gain_vs_spacing_natural_images.py
- For the analysis of learnt lateral kernels and comparision with feedforward kernels: analysis_ff_lat_ori_diff.py
- For comparing prediction strengths between models and the control network as edge strength changes: scatter_plot_model_vs_control_edge_v2.py

## Figures & Results
- Various scripts in ./misc directory can generate plots of the various experiments.
