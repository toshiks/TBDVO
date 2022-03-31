import datetime
import enum
from typing import Callable

import numpy as np
import torch

from src.models.modules.deepvo import DeepVOInterface
from src.models.modules.deepvo.layers import VOPoseConstrainQuaternionDecoder
from util_scripts.models_bench import DeepVOForBench, DeepVOReducedCNNFeaturesForBench, DeepVOTransformerEncoderForBench


def get_random_seq(seq_length: int = 7, with_cnn_model: bool = False, is_cuda: bool = True,
                   cnn_features: int = 1024) -> torch.Tensor:
    if with_cnn_model:
        dummy_input = torch.stack([
            torch.randn((1, 3, 180, 600)) for _ in range(seq_length)
        ], dim=1)

        dummy_input = torch.cat([dummy_input[:, :-1], dummy_input[:, 1:]], dim=2)
    else:
        dummy_input = torch.randn((1, seq_length - 1, cnn_features))

    if is_cuda:
        return dummy_input.cuda()

    return dummy_input


class ModelType(enum.Enum):
    Original = 0
    ReducedCNN = 1
    Transformer = 2


def get_model(model_type: ModelType, is_cuda: bool = True, seq_length: int = 7) -> DeepVOInterface:
    if model_type == ModelType.Original:
        model = DeepVOForBench(
            image_shape=(600, 180), sequence_len=seq_length, hidden_size=1000,
            pose_decoder=VOPoseConstrainQuaternionDecoder(input_dim=1000)
        )
    elif model_type == ModelType.ReducedCNN:
        model = DeepVOReducedCNNFeaturesForBench(
            hidden_size=1000,
            sequence_len=seq_length,
            pose_decoder=VOPoseConstrainQuaternionDecoder(input_dim=1000)
        )
    else:
        model = DeepVOTransformerEncoderForBench(
            hidden_size=1024,
            sequence_len=seq_length,
            pose_decoder=VOPoseConstrainQuaternionDecoder(input_dim=1024)
        )

    model = model.eval()

    if is_cuda:
        return model.cuda()

    return model


def setup_bench():
    torch.random.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.enabled = True


def run_model_with_cnn(model: DeepVOInterface, pairs: torch.Tensor):
    return model.decode(model.forward_seq(model.forward_cnn(pairs)))


def run_model_without_cnn(model: DeepVOInterface, seq: torch.Tensor):
    return model.decode(model.forward_seq(seq))


def burn_session(run_model: Callable, steps: int = 50):
    for i in range(steps):
        run_model()


def cuda_benchmark(model_type: ModelType, with_cnn_model: bool, count_loops: int = 300):
    setup_bench()
    model = get_model(model_type, is_cuda=True)
    inputs = get_random_seq(is_cuda=True, with_cnn_model=with_cnn_model,
                            cnn_features=model.cnn_feature_vector_size)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros((count_loops, 1))

    if with_cnn_model:
        run_model = lambda: run_model_with_cnn(model, inputs)
    else:
        run_model = lambda: run_model_without_cnn(model, inputs)

    with torch.no_grad():
        burn_session(run_model, count_loops // 3)

        for rep in range(count_loops):
            starter.record()
            run_model()
            ender.record()

            # wait for GPU sync
            torch.cuda.synchronize()

            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    # calculate mean and standard deviation
    mean_syn = np.sum(timings) / count_loops
    std_syn = np.std(timings)

    print(f"Cuda=True CNN={with_cnn_model} Name: {model_type.name} --- Time Mean: {mean_syn} Std: {std_syn}")


def cpu_benchmark(model_type: ModelType, with_cnn_model: bool, count_loops: int = 300):
    setup_bench()
    model = get_model(model_type, is_cuda=False)
    inputs = get_random_seq(is_cuda=False, with_cnn_model=with_cnn_model,
                            cnn_features=model.cnn_feature_vector_size)

    if with_cnn_model:
        run_model = lambda: run_model_with_cnn(model, inputs)
    else:
        run_model = lambda: run_model_without_cnn(model, inputs)

    # Init loggers
    timings = np.zeros((count_loops, 1))

    with torch.no_grad():
        burn_session(run_model, count_loops // 3)

        for rep in range(count_loops):
            start = datetime.datetime.now()
            run_model()
            end = datetime.datetime.now()

            curr_time = end - start
            timings[rep] = curr_time.total_seconds() * 1000

    # calculate mean and standard deviation
    mean_syn = np.sum(timings) / count_loops
    std_syn = np.std(timings)

    print(f"Cuda=False CNN={with_cnn_model} Name: {model_type.name} --- Time Mean: {mean_syn} Std: {std_syn}")


def run_benchmarks():
    cnn_types = [True, False]
    model_types = [ModelType.Original, ModelType.ReducedCNN, ModelType.Transformer]
    benchmarks = [cpu_benchmark, cuda_benchmark]

    for bench in benchmarks:
        for cnn_type in cnn_types:
            for model_type in model_types:
                bench(model_type, cnn_type)


if __name__ == '__main__':
    run_benchmarks()
