# Third-party notices

These notices concern third-party material included in this repository. They do not change the terms governing the rest of the repository, which is covered by `LICENSE`.

## Emergent Misalignment

Evaluation questions and judge-rubric text in `context_answer/personas.py` and `context_answer/eval/alignment.py` are attributed to Betley et al., *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs* (2025), and the `emergent-misalignment/emergent-misalignment` repository. The code here combines and adapts evaluation material into its own implementation. Upstream source links are retained in the code.

- Source: https://github.com/emergent-misalignment/emergent-misalignment
- Evaluation material: https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml
- Copyright (c) 2025 emergent-misalignment
- License: MIT. The complete upstream notice is included in `licenses/emergent-misalignment-MIT.txt`.

## Persona Vectors

Trait descriptions, generation templates, instruction pairs, evaluation questions and rubric material in `pipeline/scripts/experiment_038_common.py` are attributed to Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans and Jack Lindsey, *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* (2025), and the official `safety-research/persona_vectors` repository. That module identifies the reused artifacts and implements its own capture, judging and generation wrappers. Material is organized into Python constants. Some traits use newly generated artifacts as described in the module.

- Paper: https://arxiv.org/abs/2507.21509
- Source: https://github.com/safety-research/persona_vectors
- Generation template: https://github.com/safety-research/persona_vectors/blob/main/data_generation/prompts.py
- Example trait artifacts: https://github.com/safety-research/persona_vectors/blob/main/data_generation/trait_data_extract/evil.json
- Copyright 2025 The Persona Vectors authors
- License: Apache License 2.0. The complete upstream license is included in `licenses/persona-vectors-Apache-2.0.txt`.

## External dependencies and datasets

Libraries, model weights and external datasets referenced by the code retain their own terms. A reference or download URL does not grant permission to redistribute that resource. This repository does not contain the conversation banks, activation banks, model weights or fitted weight tensors described in `pipeline/README.md`.
