"""Two-stream seeded Transformer used by PriorPose."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2_utils

try:
    import xformers.ops as xops
except ImportError:  # xFormers is optional; PyTorch SDPA is the default fallback.
    xops = None


class SDPAMultiheadAttention(nn.Module):
    def __init__(self, dim=256, heads=4, attention_dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.attention_dropout = float(attention_dropout)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, query, key, value, additive_mask=None):
        batch, query_count, dim = query.shape
        key_count = key.shape[1]
        q = self.query(query).view(
            batch, query_count, self.heads, self.head_dim
        ).transpose(1, 2).contiguous()
        k = self.key(key).view(
            batch, key_count, self.heads, self.head_dim
        ).transpose(1, 2).contiguous()
        v = self.value(value).view(
            batch, key_count, self.heads, self.head_dim
        ).transpose(1, 2).contiguous()

        dropout = self.attention_dropout if self.training else 0.0
        if xops is not None and additive_mask is None:
            result = xops.memory_efficient_attention(
                q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), p=dropout
            ).view(batch, self.heads, query_count, self.head_dim)
        elif hasattr(F, "scaled_dot_product_attention"):
            context = nullcontext()
            if q.is_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
                deterministic = torch.are_deterministic_algorithms_enabled()
                context = torch.backends.cuda.sdp_kernel(
                    enable_flash=not deterministic,
                    enable_math=deterministic,
                    enable_mem_efficient=not deterministic,
                )
            with context:
                result = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=additive_mask,
                    dropout_p=dropout,
                    is_causal=False,
                )
        else:
            scores = q @ k.transpose(-1, -2) / self.head_dim**0.5
            if additive_mask is not None:
                scores = scores + additive_mask
            attention = scores.softmax(dim=-1)
            if dropout:
                attention = F.dropout(attention, p=dropout, training=True)
            result = attention @ v

        result = result.transpose(1, 2).contiguous().view(
            batch, query_count, dim
        )
        return self.output(result)


class FeedForward(nn.Module):
    def __init__(self, dim=256, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens):
        return self.layers(tokens)


class SeededBlock(nn.Module):
    def __init__(self, dim=256, heads=4, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.seed_input_norm = nn.LayerNorm(dim)
        self.seed_attention = SDPAMultiheadAttention(dim, heads)
        self.seed_dropout = nn.Dropout(dropout)
        self.seed_output_norm = nn.LayerNorm(dim)
        self.token_input_norm = nn.LayerNorm(dim)
        self.cross_attention = SDPAMultiheadAttention(dim, heads)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_output_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_output_norm = nn.LayerNorm(dim)

    def forward(self, seeds, tokens, additive_mask):
        normalized = self.seed_input_norm(seeds)
        update = self.seed_attention(normalized, normalized, normalized)
        seeds = self.seed_output_norm(seeds + self.seed_dropout(update))

        normalized = self.token_input_norm(tokens)
        update = self.cross_attention(
            normalized, seeds, seeds, additive_mask=additive_mask
        )
        tokens = self.cross_output_norm(tokens + self.cross_dropout(update))
        tokens = self.ffn_output_norm(
            tokens + self.ffn_dropout(self.feed_forward(tokens))
        )
        return seeds, tokens


def _fps(xyz, count):
    return pointnet2_utils.furthest_point_sample(
        xyz.contiguous(), min(int(count), xyz.shape[1])
    ).long()


def _knn_mask(query_xyz, seed_xyz, neighbors):
    neighbors = min(int(neighbors), seed_xyz.shape[1])
    distances = torch.cdist(query_xyz, seed_xyz)
    indices = distances.topk(k=neighbors, dim=-1, largest=False).indices
    keep = torch.zeros(
        *distances.shape, device=query_xyz.device, dtype=torch.bool
    )
    keep.scatter_(2, indices, True)
    mask = torch.zeros_like(distances)
    mask.masked_fill_(~keep, float("-inf"))
    return mask.unsqueeze(1)


@torch.no_grad()
def _geometry_quality(xyz, seeds, neighbors):
    neighbors = min(int(neighbors), seeds.shape[1])
    distances = torch.cdist(xyz, seeds)
    mean_distance = distances.topk(
        k=neighbors, dim=-1, largest=False
    ).values.mean(dim=-1)
    sigma = mean_distance.mean(dim=1, keepdim=True) + 1e-6
    quality = torch.exp(-0.5 * (mean_distance / sigma) ** 2).clamp(0.0, 1.0)
    return (quality / (quality.mean(dim=1, keepdim=True) + 1e-6)).clamp(0.0, 1.0)


class TwoStreamSeededTransformer(nn.Module):
    """Camera, observation and prior tokens with fixed geometry routing.

    The number of updates is fixed by ``blocks``. ``share_parameters`` selects
    a recurrent block or independently parameterized blocks. Both dataset
    configs use independent blocks and geometry-quality gating in both streams.
    """

    def __init__(
        self,
        blocks=6,
        dim=256,
        heads=4,
        hidden_dim=1024,
        observation_seeds=96,
        prior_seeds=64,
        neighbors=32,
        score_bias_gamma=0.2,
        dropout=0.0,
        share_parameters=False,
    ):
        super().__init__()
        self.update_count = int(blocks)
        self.share_parameters = bool(share_parameters)
        self.observation_seed_count = observation_seeds
        self.prior_seed_count = prior_seeds
        self.neighbors = neighbors
        self.heads = heads
        self.score_bias_gamma = score_bias_gamma
        parameterized_blocks = 1 if self.share_parameters else self.update_count
        self.blocks = nn.ModuleList(
            [SeededBlock(dim, heads, hidden_dim, dropout)
             for _ in range(parameterized_blocks)]
        )

    def _routing_bias(
        self,
        observation_xyz,
        prior_xyz,
        observation_seeds,
        prior_seeds,
        observation_quality,
        prior_quality,
        observation_seed_quality,
        prior_seed_quality,
    ):
        batch, observation_count, _ = observation_xyz.shape
        prior_count = prior_xyz.shape[1]
        obs_seed_count = observation_seeds.shape[1]
        prior_seed_count = prior_seeds.shape[1]
        token_count = 1 + observation_count + prior_count
        seed_count = 1 + obs_seed_count + prior_seed_count
        mask = torch.zeros(
            batch,
            1,
            token_count,
            seed_count,
            device=observation_xyz.device,
            dtype=torch.float32,
        )
        mask[:, :, 1 : 1 + observation_count, 1 : 1 + obs_seed_count] = (
            _knn_mask(observation_xyz, observation_seeds, self.neighbors)
        )
        mask[
            :,
            :,
            1 + observation_count :,
            1 + obs_seed_count :,
        ] = _knn_mask(prior_xyz, prior_seeds, self.neighbors)

        query_score = torch.zeros(
            batch, token_count, device=observation_xyz.device
        )
        query_score[:, 1 : 1 + observation_count] = observation_quality
        query_score[:, 1 + observation_count :] = prior_quality
        seed_score = torch.zeros(
            batch, seed_count, device=observation_xyz.device
        )
        seed_score[:, 1 : 1 + obs_seed_count] = observation_seed_quality
        seed_score[:, 1 + obs_seed_count :] = prior_seed_quality
        bias = self.score_bias_gamma * (
            query_score.unsqueeze(-1) + seed_score.unsqueeze(1)
        )
        mask = mask + bias.unsqueeze(1)
        return mask.expand(-1, self.heads, -1, -1).contiguous()

    def forward(self, tokens, observation_xyz, prior_xyz):
        batch, observation_count, _ = observation_xyz.shape
        prior_count = prior_xyz.shape[1]
        if tokens.shape[1] != 1 + observation_count + prior_count:
            raise ValueError("token and point counts do not match")

        observation_indices = _fps(
            observation_xyz, self.observation_seed_count
        )
        prior_indices = _fps(prior_xyz, self.prior_seed_count)
        batch_indices = torch.arange(batch, device=tokens.device)[:, None]

        camera = tokens[:, :1]
        observation = tokens[:, 1 : 1 + observation_count]
        prior = tokens[:, 1 + observation_count :]
        observation_seed_xyz = observation_xyz[
            batch_indices, observation_indices
        ]
        prior_seed_xyz = prior_xyz[batch_indices, prior_indices]
        observation_seed_features = observation[
            batch_indices, observation_indices
        ]
        prior_seed_features = prior[batch_indices, prior_indices]

        observation_quality = _geometry_quality(
            observation_xyz, observation_seed_xyz, self.neighbors
        )
        prior_quality = _geometry_quality(
            prior_xyz, prior_seed_xyz, self.neighbors
        )
        observation_seed_quality = observation_quality[
            batch_indices, observation_indices
        ]
        prior_seed_quality = prior_quality[batch_indices, prior_indices]
        routing_bias = self._routing_bias(
            observation_xyz,
            prior_xyz,
            observation_seed_xyz,
            prior_seed_xyz,
            observation_quality,
            prior_quality,
            observation_seed_quality,
            prior_seed_quality,
        )
        seeds = torch.cat(
            [camera, observation_seed_features, prior_seed_features], dim=1
        )

        for update_index in range(self.update_count):
            block = (
                self.blocks[0]
                if self.share_parameters
                else self.blocks[update_index]
            )
            observation = observation * observation_quality.unsqueeze(-1)
            prior = prior * prior_quality.unsqueeze(-1)
            all_tokens = torch.cat([camera, observation, prior], dim=1)
            seeds, all_tokens = block(seeds, all_tokens, routing_bias)
            camera = all_tokens[:, :1]
            observation = all_tokens[:, 1 : 1 + observation_count]
            prior = all_tokens[:, 1 + observation_count :]
        return torch.cat([camera, observation, prior], dim=1)
