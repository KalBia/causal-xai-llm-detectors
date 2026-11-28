# WEEKLY NOTES
## Weekly: 30.10.2026
Done:
  * Design Proposal made and sent to project supervisor.
  * Started broader literature review.
  * Chose technical stack.

## Weekly: 06.11.2026
Done:
  * Made progress in literature review.
  * Chose detectors, metrics, explanation methods and datasets.
  * Set up repository structure.

Plans for this week:
  * Make prototype of pipeline in repository.
  * Add detectors to repository.
  * Add datasets to repository.
  * Set up explanation methods in repository.

## Weekly: 13.11.2026
Done:
  * Made the skeleton of the project pipeline.
  * Added the data module with an external M4 submodule.

Delayed:
  * Add detectors to the repository.
  * Set up explanation methods in the repository.

Plans for this week:
  * Start detector fine-tuning.

## Weekly: 20.11.2026
Done:
 * Added detectors to the repository.
 * Updated `test_pipeline.py`.
 * Ran a small test to check the detectors (400 samples in Google Colab; full training will be on a local GPU).

Delayed:
* Set up explanation methods in the repository (dependent on detector readiness/training).

Plans for this week:
* Continue detector fine-tuning
* Set up explanation methods in the repository.
* Verify that pipeline works so far.

## Weekly: 27.11.2026

Done:
* Implemented explanation methods in the repository (SHAP Partition explainer, Integrated Gradients via Captum, and random baseline with seeded draws) with unit tests.
* Added WhiteBoxDetector protocol for gradient-based explainers and aligned it with RobertaDetector.
* Added functional run.py with tests, improved explainer test coverage, and added documentation (README + src/README).
* Merged explainers module and explain stage into main.
  
Delayed:
* Full detector fine-tuning on local GPU.
* Causal validation mechanism (perturbation scripts for token ablation and synonymization, unit tests and metrics).

Plans for this week:
* Finish full detector fine-tuning on local GPU.
* Implement perturbation scripts (token ablation and synonymization), unit tests, and faithfulness evaluation.
* Add tests for experiments.
