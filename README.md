# ORTCL (Orthogonal Rotation Transformation-based Continuous Learning)

ORTCL is a continual learning method designed for time series foundation models (TSFMs), specifically tailored for streaming data. It leverages orthogonal rotation transformations to enable knowledge transfer and mitigate catastrophic forgetting. More detailed code and documentation will be released in the future. Below are some example commands to get you started.

### Usage Examples

#### 1. **Fine-Tuning Mode**  
- Run the fine-tuning script on the **ECL dataset**:  
  ```bash
  bash ./scripts/forecast/ECL_tuning.sh
  ```

#### 2. **Model Editing Mode (via ORTCL)**  
- Run the ORTCL model editing script on the **ECL dataset**:  
  ```bash
  bash ./scripts/forecast/ECL_ORTCL.sh
  ```

### Output

- Log files are saved under the `./log/` directory.  
- Model checkpoints are saved under the `./checkpoints/` directory.