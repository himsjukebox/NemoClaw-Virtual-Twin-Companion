# Scripts

## generate_training_data.py

Generates physics-validated synthetic training data for fine-tuning an LLM to reliably translate natural-language drone design goals into correct parametric values.

### What It Does

1. Generates diverse natural-language design prompts (14 template variations × 5 airframes × 4 materials × 4 use cases)
2. Produces valid parametric outputs (arm_count, arm_length, arm_width, material_thickness, center_cutout_radius)
3. **Validates every pair through the Physics Engine** — only structurally sound, flight-capable designs are included
4. Outputs in standard JSONL format compatible with NeMo Framework fine-tuning

### Usage

```bash
# Generate 1000 pairs (default)
python scripts/generate_training_data.py

# Generate 5000 pairs with custom output path
python scripts/generate_training_data.py --count 5000 --output data/training_5k.jsonl

# Reproducible generation with specific seed
python scripts/generate_training_data.py --count 2000 --seed 123
```

### Output Format

Each line in the JSONL file:
```json
{
  "instruction": "Design a racing quadcopter chassis using carbon fiber with 0.1kg payload",
  "output": "{\"component_type\": \"chassis\", \"design_parameters\": {\"arm_count\": 4, \"arm_length\": 95.2, ...}, \"material\": \"carbon_fiber\", \"payload_mass_kg\": 0.1, \"use_case\": \"racing\"}"
}
```

### Quality Guarantee

Every output is validated to:
- ✅ Pass structural bending stress check (stress ≤ yield / safety_factor)
- ✅ Pass the arm_width ≥ arm_length × 0.08 heuristic
- ✅ Meet the use-case TWR target (racing ≥ 4.0, others ≥ 2.0)
- ✅ Have feasible payload capacity (TWR ≥ 1.0 with payload)
- ✅ All parameters within valid ranges

### Fine-Tuning Pipeline (with NVIDIA NeMo)

```bash
# 1. Generate data
python scripts/generate_training_data.py --count 5000

# 2. Fine-tune with NeMo Framework (requires DGX / GPU)
# python -m nemo.collections.nlp.models.language_modeling.megatron_gpt_sft \
#     trainer.devices=1 \
#     model.data.train_ds.file_path=data/training_data.jsonl \
#     model.data.train_ds.prompt_template="<|user|>\n{instruction}\n<|assistant|>\n{output}" \
#     ...
```

### No API Key Required

The data generation is entirely local — it uses the deterministic Physics Engine for validation, with no NVIDIA NIM or LLM calls. You can generate millions of pairs on a laptop.
