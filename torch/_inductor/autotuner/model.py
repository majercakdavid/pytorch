import copy
import itertools

import torch
import numpy as np
import regex as re
from torch import nn
from torch._inductor.dependencies import StarDep, WeakDep
from torch._inductor import config
from torch._inductor.triton_heuristics import unique_configs
from torch._inductor.virtualized import V


### model arch related
class ModelType(enumerate):
    XGB_BASELINE = 0
    NN_POINTWISE = 1
    NN_PAIRWISE = 2
    NN_PAIRWISE_SMALL = 3


class NN(nn.Module):
    def __init__(self, hidden_dim, use_norm=True, activation="tanh"):
        super().__init__()

        self.kernel_category_embedding = torch.nn.Embedding(
            num_embeddings=3, embedding_dim=32
        )
        self.num_of_loops_embedding = torch.nn.Embedding(
            num_embeddings=10, embedding_dim=32
        )

        self.hidden_dim = [576] + hidden_dim + [1]
        self.num_layers = len(self.hidden_dim) - 1

        self.op_bag_ln = nn.ModuleList([nn.Linear(1, 2) for i in range(56)])
        self.is_contiguous_ln = torch.nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.is_scalar_ln = torch.nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.is_indirect_ln = torch.nn.Embedding(num_embeddings=2, embedding_dim=4)

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
                for i in range(self.num_layers)
            ]
        )
        self.use_norm = use_norm
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim[i + 1]) for i in range(self.num_layers - 1)]
        )

        torch.nn.init.xavier_normal_(self.kernel_category_embedding.weight)
        torch.nn.init.xavier_normal_(self.num_of_loops_embedding.weight)
        torch.nn.init.xavier_normal_(self.is_contiguous_ln.weight)
        torch.nn.init.xavier_normal_(self.is_scalar_ln.weight)
        torch.nn.init.xavier_normal_(self.is_indirect_ln.weight)
        for layer in list(self.op_bag_ln) + list(self.layers):
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        if activation == "tanh":
            self.activation = torch.nn.functional.tanh
        elif activation == "leaky_relu":
            self.activation = torch.nn.functional.leaky_relu
        else:
            assert False, "Unknown activation"

    def forward(self, x):
        x = torch.cat(
            [
                self.kernel_category_embedding(x[:, 0].long()),
                self.num_of_loops_embedding(x[:, 1].long()),
                torch.cat(
                    [self.op_bag_ln[i](x[:, 2 + i].unsqueeze(1)) for i in range(56)],
                    dim=1,
                ),
                x[:, 58:60],
                torch.cat(
                    [
                        torch.cat(
                            [
                                x[:, 60 + i * 17 : 60 + i * 17 + 14],
                                torch.cat(
                                    [
                                        self.is_contiguous_ln(
                                            x[:, 60 + i * 17 + 14].long()
                                        ),
                                        self.is_scalar_ln(
                                            x[:, 60 + i * 17 + 15].long()
                                        ),
                                        self.is_indirect_ln(
                                            x[:, 60 + i * 17 + 16].long()
                                        ),
                                    ],
                                    dim=1,
                                ),
                            ],
                            dim=1,
                        )
                        for i in range(15)
                    ],
                    dim=1,
                ),
                x[:, 315:],
            ],
            dim=1,
        )
        if self.use_norm:
            for norm, layer in zip(self.norms, self.layers[:-1]):
                x = self.activation(norm(layer(x)))
        else:
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


def get_model(model_type: ModelType):
    if model_type == ModelType.XGB_BASELINE:
        import xgboost

        return xgboost.XGBRegressor(
            max_depth=15,
            learning_rate=0.2,
            n_estimators=120,
            tree_method="hist",
            predictor="cpu_predictor",
            eval_metric=["rmse", "mae"],
        )
    elif model_type == ModelType.NN_POINTWISE:
        return NN(hidden_dim=[8192, 2048, 32])
    elif model_type == ModelType.NN_PAIRWISE:
        return NN(hidden_dim=[4096, 1024, 32])
    elif model_type == ModelType.NN_PAIRWISE_SMALL:
        return NN(hidden_dim=[8192, 64], use_norm=False, activation="leaky_relu")
    else:
        assert False, "Unknown model type"


### feature extraction related

# op_dict needs to be deterministic
op_dict = {
    "load": 0,
    "to_dtype": 1,
    "add": 2,
    "reduction": 3,
    "constant": 4,
    "div": 5,
    "store": 6,
    "sub": 7,
    "square": 8,
    "rsqrt": 9,
    "mul": 10,
    "tanh": 11,
    "ne": 12,
    "where": 13,
    "indirect_indexing": 14,
    "log": 15,
    "neg": 16,
    "exp": 17,
    "maximum": 18,
    "minimum": 19,
    "index_expr": 20,
    "ge": 21,
    "masked": 22,
    "lt": 23,
    "and_": 24,
    "erf": 25,
    "eq": 26,
    "le": 27,
    "gt": 28,
    "relu": 29,
    "sqrt": 30,
    "logical_not": 31,
    "load_seed": 32,
    "rand": 33,
    "abs": 34,
    "reciprocal": 35,
    "ceil": 36,
    "sigmoid": 37,
    "sin": 38,
    "cos": 39,
    "logical_and": 40,
    "bitwise_and": 41,
    "randn": 42,
    "floor": 43,
    "remainder": 44,
    "isinf": 45,
    "logical_or": 46,
    "expm1": 47,
    "libdevice_sqrt": 48,
    "libdevice_log": 49,
    "truediv": 50,
    "sign": 51,
    "randint64": 52,
    "bitwise_or": 53,
    "pow": 54,
    "isnan": 55,
}


class KernelCategory(enumerate):
    POINTWISE = 0
    REDUCTION = 1
    PERSISTENT_REDUCTION = 2


def get_kernel_category(src: str) -> KernelCategory:
    if "@pointwise" in src:
        return KernelCategory.POINTWISE
    if "@reduction" in src:
        return KernelCategory.REDUCTION
    if "@persistent_reduction" in src:
        return KernelCategory.PERSISTENT_REDUCTION


def get_number_of_loops(src: str) -> int:
    return src.count("for roffset in range(0, rnumel, RBLOCK):")


def parse_list_of_numbers(s: str) -> list:
    # num1, num2, num3, ...
    nums = s.strip().split(",")
    nums = [num.strip() for num in nums]
    return [int(num) for num in nums]


def get_size_hints(src: str) -> list:
    return parse_list_of_numbers(re.search(r"size_hints=\[([^\]]*)\]", src).group(1))


def get_tiling(src: str) -> list:
    names = ["xnumel", "ynumel", "rnumel"]
    result = list()
    for name in names:
        startpos = src.find(name + " =")
        if startpos == -1:
            result.append(1)
            continue
        endpos = src.find("\n", startpos)
        result.append(int(src[startpos + len(name + " = ") : endpos]))
    return result


def pad_tensor():
    tensor_feature = list()
    tensor_feature.append(False)  # StarDepOrWeakDep
    tensor_feature.append(0)  # bytes
    # we use the lowest 6 dims of the tensor
    tensor_feature.extend([0] * 6)  # strides,
    # we use the lowest 6 dims of the tensor
    tensor_feature.extend([0] * 6)  # size
    tensor_feature.append(True)  # is_contiguous
    tensor_feature.append(False)  # is_scalar
    tensor_feature.append(False)  # is_indirect
    return tensor_feature


def tensor_list(deps, total_bytes, rw_len):
    rw_list = list(
        [
            (dep, bytes)
            for dep, bytes in zip(deps, total_bytes)
            if not isinstance(dep, (StarDep, WeakDep))
        ]
    )
    res = list()
    # sort the tensors by bytes in descending order
    rw_list = sorted(rw_list, key=lambda x: x[1], reverse=True)
    # for dep, strides, sizes, bytes in rw_list[:rw_len]:
    for dep, bytes in rw_list[:rw_len]:
        tensor_feature = pad_tensor()
        tensor_feature[0] = isinstance(dep, (StarDep, WeakDep))
        tensor_feature[1] = bytes
        # left pad strides, strides can be None if StarDep or WeakDep
        # if strides is not None:
        strides = V.graph.sizevars.stride_hints(dep.index, dep.var_names)
        for i in range(len(strides)):
            # we use the lowest 6 dims of the tensor
            tensor_feature[8 - (len(strides) - i)] = strides[i]
        # left pad size, sizes can be None if StarDep or WeakDep
        # if sizes is not None:
        sizes = [int(size_) for size_ in dep.size]
        for i in range(len(sizes)):
            # we use the lowest 6 dims of the tensor
            tensor_feature[14 - (len(sizes) - i)] = sizes[i]
        tensor_feature[-3] = dep.is_contiguous()
        tensor_feature[-2] = dep.is_scalar()
        tensor_feature[-1] = dep.is_indirect()

        # if strides is not None and sizes is not None:
        #     assert len(strides) == len(sizes)
        #     for size_ in sizes:
        #         assert isinstance(size_, int)
        #     for stride in strides:
        #         assert isinstance(stride, int)
        assert len(dep.size) == len(strides)
        for size_ in dep.size:
            assert size_.is_integer
        for stride in strides:
            assert isinstance(stride, int)

        res.append(tensor_feature)
    # right pad with empty tensor
    for i in range(rw_len - len(rw_list)):
        res.append(pad_tensor())
    return res


### search space related
class AutotunerSpaceCategory(enumerate):
    MAX_AUTOTUNE_TOP1 = 0
    MAX_AUTOTUNE_TOP2 = 1
    RADIUS_1_TOP1 = 2
    RADIUS_1_TOP2 = 3


# This class is inherited from coordinate_descent_tuner.py
class SearchSpaceGenerator:
    def __init__(self, size_hints):
        self.size_hints = size_hints

    def get_xmax(self):
        xmax = config.triton.max_block["X"]
        if self.size_hints and len(self.size_hints) > 0:
            xmax = min(xmax, self.size_hints[0])
        return xmax

    def get_ymax(self):
        ymax = config.triton.max_block["Y"]
        if self.size_hints and len(self.size_hints) > 1:
            ymax = min(ymax, self.size_hints[1])
        return ymax

    def get_zmax(self):
        zmax = config.triton.max_block["Z"]
        if self.size_hints and len(self.size_hints) > 2:
            zmax = min(zmax, self.size_hints[2])
        return zmax

    def get_rmax(self):
        if self.size_hints and len(self.size_hints) > 0:
            return self.size_hints[-1]  # the last one is for reduction
        else:
            # large enough. We should not pick this large RBLOCK anyway
            return 2**30

    @property
    def tunable_fields(self):
        out = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # NOTE: we should not tune RBLOCK for persistent reduction.
            # We rely on the fact that persistent reduction's triton.Config
            # does not have the RBLOCK field to guarantee that.
            "RBLOCK",
            # the following 3 are for mm
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        return out

    def value_too_large(self, name, val):
        if name == "XBLOCK":
            return val > self.get_xmax()
        if name == "YBLOCK":
            return val > self.get_ymax()
        if name == "ZBLOCK":
            return val > self.get_zmax()
        if name == "RBLOCK":
            return val > self.get_rmax()

        return False

    def get_neighbour_values(self, name, orig_val, radius=1, include_self=False):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as it's own neighbour.
        """
        assert radius >= 1

        def update(cur_val, inc=True):
            if name == "num_stages":
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            else:
                if inc:
                    return cur_val * 2
                else:
                    return cur_val // 2

        out = []
        # increment loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)

        # decrement loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if cur_val <= 0:
                break
            out.append(cur_val)

        if include_self:
            out.append(orig_val)
        return out

    def get_field(self, config, name):
        if name == "num_warps":
            return config.num_warps
        elif name == "num_stages":
            return config.num_stages
        else:
            return config.kwargs.get(name, None)

    def set_field(self, config, name, value):
        if name == "num_warps":
            config.num_warps = value
        elif name == "num_stages":
            config.num_stages = value
        else:
            config.kwargs[name] = value

    def check_all_tuning_directions(self, st_config, radius):
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = self.get_field(st_config, field)
            if old_value is None:
                continue
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=radius,
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        res = list()
        choices = itertools.product(*candidate_values_list)
        for choice in choices:
            assert len(choice) == len(effective_fields)
            candidate_config = copy.deepcopy(st_config)
            for new_val, field in zip(choice, effective_fields):
                self.set_field(candidate_config, field, new_val)
            res.append(candidate_config)
        return res

    def generate(self, configs, radius):
        res = list()
        for config in configs:
            res.extend(self.check_all_tuning_directions(config, radius))
        return res


class AutotunerModel:
    model_type: ModelType

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = get_model(model_type)
        self.input_tensor_lim = 10
        self.output_tensor_lim = 5

    def load(self, path):
        if self.model_type == ModelType.XGB_BASELINE:
            self.model.load_model(path)
        else:
            self.model.load_state_dict(torch.load(path))

    def prepare(self):
        if self.model_type != ModelType.XGB_BASELINE:
            self.model = self.model.to("cuda")
            self.model.eval()

    def score(self, configs, autotuner_raw_data):
        X = self.get_feature_vec(configs, autotuner_raw_data)
        if self.model_type == ModelType.XGB_BASELINE:
            scores = self.model.predict(X) * -1
        else:
            X = torch.from_numpy(X).to("cuda")
            scores = self.model.forward(X).squeeze().cpu().detach().numpy()
            if self.model_type == ModelType.NN_POINTWISE:
                scores = scores * -1
        indices = np.argsort(scores)
        return [configs[i] for i in indices]

    def predict(self, configs, autotuner_raw_data, autotuner_space):
        _, _, src_code, _ = autotuner_raw_data
        size_hints = get_size_hints(src_code)

        configs = unique_configs(configs)
        if autotuner_space in [
            AutotunerSpaceCategory.RADIUS_1_TOP1,
            AutotunerSpaceCategory.RADIUS_1_TOP2,
        ]:
            configs = unique_configs(
                SearchSpaceGenerator(size_hints).generate(configs, radius=1)
            )

        configs = self.score(configs, autotuner_raw_data)
        if autotuner_space in [
            AutotunerSpaceCategory.MAX_AUTOTUNE_TOP1,
            AutotunerSpaceCategory.RADIUS_1_TOP1,
        ]:
            return configs[:1]
        elif autotuner_space in [
            AutotunerSpaceCategory.MAX_AUTOTUNE_TOP2,
            AutotunerSpaceCategory.RADIUS_1_TOP2,
        ]:
            return configs[:2]
        else:
            assert False, "Unknown autotuner space"

    def get_feature_vec(self, configs, autotuner_raw_data):
        (
            # (reads, writes, strides, sizes, total_bytes),
            (reads, writes, total_bytes),
            node_read_writes,
            src_code,
            autotuner_dict,
        ) = autotuner_raw_data

        # Get the kernel category
        kernel_category = get_kernel_category(src_code)
        if kernel_category is None:
            return None

        # Get the number of loops
        if kernel_category is KernelCategory.REDUCTION:
            num_of_loops = get_number_of_loops(src_code)
        else:
            num_of_loops = 0

        # Get the op dict
        op_counts = node_read_writes.op_counts
        op_bag = dict()
        for op in sorted(op_counts.keys()):
            assert op in op_dict, "Unknown op: " + op
            op_bag[op_dict[op]] = op_counts[op]

        # Get the tilings
        numels = get_tiling(src_code)

        # Get the size hints
        size_hints = get_size_hints(src_code)
        assert size_hints == autotuner_dict["size_hints"]

        # Get the input tensors and output tensors
        read_deps = tensor_list(
            reads,
            # strides[: len(reads)],
            # sizes[: len(reads)],
            total_bytes[: len(reads)],
            self.input_tensor_lim,
        )
        write_deps = tensor_list(
            writes,
            # strides[len(reads) :],
            # sizes[len(reads) :],
            total_bytes[len(reads) :],
            self.output_tensor_lim,
        )

        X = list()
        for config in configs:
            feature_vector = list()
            feature_vector.append(kernel_category)
            feature_vector.append(num_of_loops)
            op_bag_vec = [0] * len(op_dict)
            for op in op_bag:
                op_bag_vec[op] = op_bag[op]
            feature_vector.extend(op_bag_vec)
            size_hints_vec = [1] * 2
            for i in range(len(size_hints)):
                size_hints_vec[i] = size_hints[i]
            feature_vector.extend(size_hints_vec)

            for tensor in read_deps:
                feature_vector.extend(tensor)
            for tensor in write_deps:
                feature_vector.extend(tensor)

            if "XBLOCK" in config.kwargs:
                feature_vector.append(config.kwargs["XBLOCK"])
            else:
                feature_vector.append(1)
            if "YBLOCK" in config.kwargs:
                feature_vector.append(config.kwargs["YBLOCK"])
            else:
                feature_vector.append(1)
            if "RBLOCK" in config.kwargs:
                feature_vector.append(config.kwargs["RBLOCK"])
            else:
                feature_vector.append(1)

            feature_vector.append(config.num_warps)
            feature_vector.append(config.num_stages)
            feature_vector.append(numels[0])
            feature_vector.append(numels[1])
            feature_vector.append(numels[2])

            X.append(feature_vector)

        X = np.array(X)

        if self.model_type != ModelType.XGB_BASELINE:
            # num_of_loops is embedded
            X[:, 1][X[:, 1] > 9] = 9

            X = X.astype(np.float32)
            X[:, 58:60] = np.log2(X[:, 58:60] + 1)  # size_hints
            X[:, 315] = np.log2(X[:, 315] + 1)  # XBLOCK
            X[:, 316] = np.log2(X[:, 316] + 1)  # YBLOCK
            X[:, 317] = np.log2(X[:, 317] + 1)  # RBLOCK
            X[:, 320] = np.log2(X[:, 320] + 1)  # numels[0]
            X[:, 321] = np.log2(X[:, 321] + 1)  # numels[1]
            X[:, 322] = np.log2(X[:, 322] + 1)  # numels[2]

            for i in range(10):
                base = 60 + i * 17
                X[:, base + 1] = np.log2(X[:, base + 1] + 1)  # bytes
                X[:, base + 2 : base + 8] = np.log2(  # strides
                    np.abs(X[:, base + 2 : base + 8]) + 1
                ) * np.sign(X[:, base + 2 : base + 8])
                X[:, base + 8 : base + 14] = np.log2(  # sizes
                    np.abs(X[:, base + 8 : base + 14]) + 1
                ) * np.sign(X[:, base + 8 : base + 14])

            for i in range(5):
                base = 230 + i * 17
                X[:, base + 1] = np.log2(X[:, base + 1] + 1)  # bytes
                X[:, base + 2 : base + 8] = np.log2(  # strides
                    np.abs(X[:, base + 2 : base + 8]) + 1
                ) * np.sign(X[:, base + 2 : base + 8])
                X[:, base + 8 : base + 14] = np.log2(  # sizes
                    np.abs(X[:, base + 8 : base + 14]) + 1
                ) * np.sign(X[:, base + 8 : base + 14])

        return np.array(X)
