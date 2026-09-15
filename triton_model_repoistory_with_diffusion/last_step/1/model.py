"""Triton python backend: final audio reshape.

The generator emits [batch, 1, samples]; clients want a flat float32 waveform.
That is the whole job — kept as a separate step so the ensemble's output shape
does not depend on the vocoder's.
"""

from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_OUT")
                if tensor is None:
                    raise ValueError("AUDIO_OUT is required")

                audio = np.asarray(tensor.as_numpy(), dtype=np.float32).squeeze()
                if audio.ndim == 0:
                    audio = audio.reshape(1)

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("AUDIO", np.ascontiguousarray(audio))]
                ))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"last_step failed: {exc}"),
                ))

        return responses

    def finalize(self):
        pass
