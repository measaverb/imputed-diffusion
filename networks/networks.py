import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def transformer(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def conv1d(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.proj_1 = nn.Linear(embedding_dim, projection_dim)
        self.proj_2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, t):
        x = self.embedding[t]
        x = self.proj_1(x)
        x = F.silu(x)
        x = self.proj_2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T, 1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1, dim)
        table = steps * frequencies  # (T, dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T, dim * 2)
        return table


class DiffusionBackbone(nn.Module):
    def __init__(self, config, input_dim=2):
        super().__init__()
        self.channels = config["diffusion"]["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["diffusion"]["num_steps"],
            embedding_dim=config["diffusion"]["embedding_dim"],
        )
        self.strategy_embedding = nn.Embedding(2, config["diffusion"]["embedding_dim"])

        self.input_projection = conv1d(input_dim, self.channels, 1)
        self.output_projection1 = conv1d(self.channels, self.channels, 1)
        self.output_projection2 = conv1d(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["diffusion"]["side_dim"],
                    channels=self.channels,
                    embedding_dim=config["diffusion"]["embedding_dim"],
                    nheads=config["diffusion"]["num_heads"],
                )
                for _ in range(config["diffusion"]["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, strategy_type):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        strategy_emb = self.strategy_embedding(strategy_type)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, strategy_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B, channel, K * L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B, 1, K * L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(embedding_dim, channels)
        self.strategy_projection = nn.Linear(embedding_dim, channels)

        self.cond_projection = conv1d(side_dim, 2 * channels, 1)
        self.mid_projection = conv1d(channels, 2 * channels, 1)
        self.output_projection = conv1d(channels, 2 * channels, 1)

        self.time_layer = transformer(heads=nheads, layers=1, channels=channels)
        self.feature_layer = transformer(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, strategy_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B, channel, 1)
        strategy_emb = self.strategy_projection(strategy_emb).unsqueeze(-1)

        y = x + diffusion_emb + strategy_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B, channel, K * L)
        y = self.mid_projection(y)  # (B, 2 * channel, K * L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B, 2 * channel, K * L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, channel, K * L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip