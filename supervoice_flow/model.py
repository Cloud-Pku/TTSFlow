import math
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint

from .transformer import Transformer, ConvPositionEmbed
from .tensors import drop_using_mask, merge_mask

from torch.distributions import Independent, Normal

class AudioFlow(torch.nn.Module):
    def __init__(self, config, *, cache_alibi = False):
        super(AudioFlow, self).__init__()
        self.config = config.model

        # Transformer input
        self.transformer_input = torch.nn.Linear(2 * config.audio.n_mels, self.config.n_dim)

        # Sinusoidal positional embedding for time
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(self.config.n_dim)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = self.config.n_dim, kernel_size = 31)

        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.n_heads,
            n_layers = self.config.n_layers,
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 1, # Exclude time embedding from attention bias
            att_dropout = 0,
            ffn_dropout = 0.1,
            cache_alibi = cache_alibi
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.config.n_dim, config.audio.n_mels)

    def sample(self, *, audio, mask = None, steps, alpha = None, return_trajectory = False):
        
        #
        # Prepare
        #

        # Mask out audio
        source_audio = audio
        if mask is not None:
            audio = drop_using_mask(source = audio, replacement = 0, mask = mask)

        # Create noise
        noise = torch.randn_like(audio)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = audio.device)

        #
        # Solver
        # 

        # Overwrite audio segment with predicted audio according to mask
        def merge_predicted(predicted):
            if mask is None:
                return predicted
            return merge_mask(source = source_audio, replacement = predicted, mask = mask)

        def solver(t, z):

            # If alpha is not provided
            if alpha is None:
                return self.forward(audio = audio.unsqueeze(0), noise = z.unsqueeze(0), times = t.unsqueeze(0)).squeeze(0)

            # If alpha is provided - zero out tokens and audio and mix together
            audio_empty = torch.zeros_like(audio)

            # Mix together
            audio_t = torch.stack([audio_empty, audio], dim = 0)
            noise_t = torch.stack([z, z], dim = 0) # Just double it
            t_t = torch.stack([t, t], dim = 0) # Just double it

            # Inference
            predicted_mix = self.forward(
                audio = audio_t, 
                noise = noise_t, 
                times = t_t
            )
            predicted_conditioned = predicted_mix[1]
            predicted_unconditioned = predicted_mix[0]
            
            # CFG prediction

            # There are different ways to do CFG, this is my very naive version, which worked for me:
            # prediction = (1 + alpha) * predicted_conditioned - alpha * predicted_unconditioned

            # Original paper uses a different one, but i found that it simply creates overexposed values
            # prediction = predicted_unconditioned + (predicted_conditioned - predicted_unconditioned) * alpha

            # This is from the latest paper that rescales original formula (https://arxiv.org/abs/2305.08891):
            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())

            return prediction

        with torch.no_grad():
            trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        #
        # Output sample and full trajectory
        #

        return merge_predicted(trajectory[-1]), trajectory

    def logp(self, audio, using_Hutchinson_trace_estimator=True):

        condition_audio = torch.zeros_like(audio)
        model_drift = lambda t, x: - self.forward(audio = condition_audio, noise = x, times = 1 - t)

        def compute_trace_of_jacobian_general(dx, x):
            # Assuming x has shape (B, D1, ..., Dn)
            shape = x.shape[1:]  # get the shape of a single element in the batch
            outputs = torch.zeros(
                x.shape[0], device=x.device, dtype=x.dtype
            )  # trace for each batch
            # Iterate through each index in the product of dimensions
            for index in torch.cartesian_prod(*(torch.arange(s) for s in shape)):
                if len(index.shape) > 0:
                    index = tuple(index)
                else:
                    index = (index,)
                grad_outputs = torch.zeros_like(x)
                grad_outputs[(slice(None), *index)] = (
                    1  # set one at the specific index across all batches
                )
                grads = torch.autograd.grad(
                    outputs=dx, inputs=x, grad_outputs=grad_outputs, retain_graph=True
                )[0]
                outputs += grads[(slice(None), *index)]
            return outputs

        def compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x, eps):
            """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

            fn_eps = torch.sum(dx * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True)[0]
            outputs = torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
            return outputs

        def composite_drift(t, x):
            # where x is actually x0_and_diff_logp, (x0, diff_logp), which is a tuple containing x and logp_xt_minus_logp_x0
            with torch.set_grad_enabled(True):
                t = t.detach()
                x_t = x[0].detach()
                logp_xt_minus_logp_x0 = x[1]

                x_t.requires_grad = True
                t.requires_grad = True

                dx = model_drift(t, x_t)
                if using_Hutchinson_trace_estimator:
                    noise = torch.randn_like(x_t, device=x_t.device)
                    logp_drift = -compute_trace_of_jacobian_by_Hutchinson_Skilling(
                        dx, x_t, noise
                    )
                    # logp_drift = - divergence_approx(dx, x_t, noise)
                else:
                    logp_drift = -compute_trace_of_jacobian_general(dx, x_t)

                return dx, logp_drift

        # Create time interpolation
        times = torch.linspace(0.0, 1.0, 100, device = audio.device)

        x0_and_diff_logp = (audio, torch.zeros(audio.shape[0], device=audio.device))

        def forward_ode_drift_by_torchdiffeq(t, x):
            # broadcasting t to match the batch size of x
            t = t.repeat(x[0].shape[0])
            return composite_drift(t, x)

        x1_and_logp1 = odeint(
                func=forward_ode_drift_by_torchdiffeq,
                y0=x0_and_diff_logp,
                t=times,
                atol = 1e-5, rtol = 1e-5, method = 'midpoint'
            )

        logp_x1_minus_logp_x0 = x1_and_logp1[1][-1]
        x1 = x1_and_logp1[0][-1]
        x1_1d = x1.reshape(x1.shape[0], -1)
        logp_x1 = Independent(
            Normal(
                loc=torch.zeros_like(x1_1d, device=x1_1d.device),
                scale=torch.ones_like(x1_1d, device=x1_1d.device),
            ),
            1,
        ).log_prob(x1_1d)

        log_likelihood = logp_x1 - logp_x1_minus_logp_x0

        return log_likelihood

    def forward(self, *,  
        
        # Audio
        audio, 
        noise, 

        # Extra conditioning for fine-tuning
        condition = None,
        
        # Time
        times, 

        # Training    
        mask = None,
        target = None,
        mask_loss = False
    ):
        
        #
        # Prepare
        #

        if mask is None and target is not None and mask_loss:
            raise ValueError('Mask is required when target is provided and mask_loss enabled')
        if target is None and mask is not None:
            raise ValueError('Mask is not required when target is not provided')
        if condition is not None:
            assert condition.shape[0] == audio.shape[0], 'Condition should have the same batch size as audio'
            assert condition.shape[1] == audio.shape[1], 'Condition should have the same sequence length as audio'
            assert condition.shape[2] == self.config.n_dim, 'Condition should have ' + self.config.n_dim + ' channels'

        # Check shapes
        assert audio.shape[0] == noise.shape[0] # Batch
        assert audio.shape[1] == noise.shape[1] # Sequence length
        assert audio.shape[2] == noise.shape[2] # Channels length
        if mask is not None:
            assert audio.shape[0] == mask.shape[0] # Batch
            assert audio.shape[1] == mask.shape[1] # Squence length

        #
        # Compute
        #

        # Combine phoneme embeddings, masked audio and noizy audio
        output = torch.cat([audio, noise], dim = -1)

        # Apply transformer input layer
        output = self.transformer_input(output)

        # Apply condition after transformer input
        if condition is not None:
            output = output + condition

        # Apply sinusoidal positional embedding
        sinu_times = self.sinu_pos_emb(times).unsqueeze(1)
        output = torch.cat([output, sinu_times], dim=1)

        # Apply convolutional positional encoder
        output = self.conv_embed(output) + output

        # Run through transformer
        output = self.transformer(output)

        # Predict durations
        output = self.prediction(output)

        # Cut to length
        output = output[:, :-1, :]

        #
        # Loss
        #

        if target is not None:
            
            # Compute MSE loss
            loss = F.mse_loss(output, target, reduction = 'none')

            # Mean for each frame
            loss = reduce(loss, 'b n d -> b n', 'mean')

            # Mask out non target frames
            if mask_loss:
                loss = loss.masked_fill(~mask, 0.)

                # Number of masked frames
                n_masked_frames = mask.sum(dim = -1).clamp(min = 1)

                # Mean loss of expectation over masked loss
                loss = loss.sum(dim = -1) / n_masked_frames
            else:
                # Mean loss of expectation over each frame
                loss = loss.sum(dim = -1) / target.shape[1]

            # Expectation over loss of batch
            loss = loss.mean()

            return output, loss
        else:
            return output


class LearnedSinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered