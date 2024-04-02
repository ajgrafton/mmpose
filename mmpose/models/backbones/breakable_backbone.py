from typing import Union, List
from torch import Tensor


class BreakableBackbone:
    def forward(self, x, part: int):
        raise NotImplementedError()

    def prep_inputs(self, inputs: Union[List[Tensor], List[List[Tensor]]]):
        raise NotImplementedError()

    @staticmethod
    def prep_output(output) -> Tensor:
        raise NotImplementedError()

    def num_output_channels(self) -> int:
        raise NotImplementedError()
