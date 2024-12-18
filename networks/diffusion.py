import numpy as np
import torch
import torch.nn as nn

from networks.networks import DiffusionBackbone


class CSDI(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config

        self.ddim_eta = 1
        self.target_dim = config["data"]["features"]

        self.emb_time_dim = config["networks"]["time_embedding"]
        self.emb_feature_dim = config["networks"]["feature_embedding"]
        self.masking_strategy = config["networks"]["masking_strategy"]
        self.masking_ratio = config["training"]["masking_ratio"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config["diffusion"]["side_dim"] = self.emb_total_dim
        input_dim = 1
        self.diffusion_backbone = DiffusionBackbone(config=config, input_dim=input_dim)

        self.num_steps = config["diffusion"]["num_steps"]
        if config["diffusion"]["scheduler"] == "quad":
            self.beta = (
                np.linspace(
                    config["diffusion"]["beta_start"] ** 0.5,
                    config["diffusion"]["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config["diffusion"]["scheduler"] == "linear":
            self.beta = np.linspace(
                config["diffusion"]["beta_start"],
                config["diffusion"]["beta_end"],
                self.num_steps,
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_tensor = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def _process_batch(self, data):
        observed_data = data["observed_data"].to(self.device).float()
        observed_mask = data["observed_mask"].to(self.device).float()
        observed_timepoint = data["timepoints"].to(self.device).float()
        gt_mask = data["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        strategy_type = data["strategy_type"].to(self.device).long()

        return (
            observed_data,
            observed_mask,
            observed_timepoint,
            gt_mask,
            for_pattern_mask,
            cut_length,
            strategy_type,
        )

    def _create_random_mask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)  # (b, *)
        for i in range(len(observed_mask)):
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * self.masking_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        conditioned_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return conditioned_mask

    def _create_historical_mask(self, observed_mask, for_pattern_mask):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.masking_strategy == "mix":
            random_mask = self._create_random_mask(observed_mask)
        conditioned_mask = observed_mask.clone()
        for i in range(len(conditioned_mask)):
            mask_choice = np.random.rand()
            if self.masking_strategy == "mix" and mask_choice > 0.5:
                conditioned_mask[i] = random_mask[i]
            else:
                conditioned_mask[i] = conditioned_mask[i] * for_pattern_mask[i - 1]
        return conditioned_mask

    def _time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _get_auxiliary_info(self, observed_timepoint, conditioned_mask):
        B, K, L = conditioned_mask.shape

        time_embed = self._time_embedding(
            observed_timepoint, self.emb_time_dim
        )  # (B, L, emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        aux_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B, L, K, *)
        aux_info = aux_info.permute(0, 3, 2, 1)  # (B, *, K, L)

        return aux_info

    def _calculate_loss(
        self,
        observed_data,
        conditioned_mask,
        observed_mask,
        aux_info,
        train,
        strategy_type,
        set_t=-1,
    ):
        B, K, L = observed_data.shape
        if not train:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_tensor[t]  # (B, 1, 1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        total_input = noisy_data.unsqueeze(1)  # (B, 1, K, L)

        prediction = self.diffusion_backbone(total_input, aux_info, t, strategy_type)
        target_mask = observed_mask - conditioned_mask

        residual = (prediction - observed_data) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def _calculate_cumulative_loss(
        self,
        observed_data,
        conditioned_mask,
        observed_mask,
        aux_info,
        train,
        strategy_type,
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self._calculate_loss(
                observed_data,
                conditioned_mask,
                observed_mask,
                aux_info,
                train,
                strategy_type,
                set_t=t,
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def forward(self, data, train=False):
        (
            observed_data,
            observed_mask,
            observed_timepoint,
            gt_mask,
            for_pattern_mask,
            _,  # cut_length
            strategy_type,
        ) = self._process_batch(data)

        if not train:
            conditioned_mask = gt_mask
        elif self.masking_strategy != "random":
            conditioned_mask = self._create_historical_mask(
                observed_mask, for_pattern_mask
            )
        else:
            conditioned_mask = self._create_random_mask(observed_mask)

        aux_info = self._get_auxiliary_info(observed_timepoint, conditioned_mask)
        loss_function = (
            self._calculate_loss if train else self._calculate_cumulative_loss
        )
        loss = loss_function(
            observed_data,
            conditioned_mask,
            observed_mask,
            aux_info,
            train,
            strategy_type,
        )
        return loss

    def _impute(
        self, observed_data, conditioned_mask, aux_info, n_samples, strategy_type
    ):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):
            noisy_obs = observed_data
            noisy_cond_history = []
            for t in range(self.num_steps):
                noise = torch.randn_like(noisy_obs)
                noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                    t
                ] ** 0.5 * noise
                noisy_cond_history.append(noisy_obs * conditioned_mask)
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                diffusion_input = (
                    conditioned_mask * noisy_cond_history[t]
                    + (1.0 - conditioned_mask) * current_sample
                )
                diffusion_input = diffusion_input.unsqueeze(1)  # (B, 1, K, L)
                predicted = self.diffusion_backbone(
                    diffusion_input,
                    aux_info,
                    torch.tensor([t]).to(self.device),
                    strategy_type,
                )

                coeff_1 = 1 / self.alpha_hat[t] ** 0.5
                coeff_2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff_1 * (current_sample - coeff_2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def evaluate(self, data, n_samples):
        (
            observed_data,
            observed_mask,
            observed_timepoint,
            gt_mask,
            _,  # for_pattern_mask
            cut_length,
            strategy_type,
        ) = self._process_batch(data)

        with torch.no_grad():
            conditioned_mask = gt_mask
            target_mask = observed_mask - conditioned_mask
            aux_info = self._get_auxiliary_info(observed_timepoint, conditioned_mask)
            samples = self._impute(
                observed_data, conditioned_mask, aux_info, n_samples, strategy_type
            )
            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
            return (
                samples,
                observed_data,
                target_mask,
                observed_mask,
                observed_timepoint,
            )
