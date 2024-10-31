from typing import Type

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


class LayerScale(nn.Module):
    initial_value: float = 1.0

    @nn.compact
    def __call__(self, x):
        gamma = self.param(
            "gamma",
            lambda _, shape: self.initial_value * jnp.ones(shape),
            (x.shape[-1],),
        )
        return x * gamma


class DropPath(nn.Module):
    rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        if self.rate > 0.0 and not deterministic:
            keep_prob = 1.0 - self.rate
            shape = (x.shape[0], 1, 1, 1)
            random_tensor = jax.random.bernoulli(
                self.make_rng("dropout"), keep_prob, shape=shape
            )
            return x / keep_prob * random_tensor
        else:
            return x


class Mlp(nn.Module):
    hidden_features: int = 1536
    out_features: int = 384
    act_layer: nn.Module = nn.gelu
    dropout_rate: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(features=self.hidden_features, use_bias=self.bias, name="fc1")(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.dropout_rate, name="drop1")(
            x, deterministic=not training
        )
        x = nn.Dense(features=self.out_features, use_bias=self.bias, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout_rate, name="drop2")(
            x, deterministic=not training
        )
        return x


class Attention(nn.Module):
    num_heads: int = 8
    attn_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_bias: bool = True
    proj_drop_rate: float = 0.0
    embed_dim: int = 384

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, N, C = x.shape
        assert (
            C == self.embed_dim
        ), f"Input embedding dimension ({C}) should match layer embedding dimension ({self.embed_dim})."
        qkv = nn.Dense(features=3 * C, use_bias=self.attn_bias, name="qkv")(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)

        # Attention matrix: (B, H, N, N)
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(C // self.num_heads)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop_rate, name="attn_drop")(
            attn, deterministic=not training
        )

        # Output: (B, N, H, C // H)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)

        x = nn.Dense(features=C, use_bias=self.proj_bias, name="proj")(x)
        x = nn.Dropout(rate=self.proj_drop_rate, name="proj_drop")(
            x, deterministic=not training
        )

        return x


class Block(nn.Module):
    num_heads: int = 6
    embed_dim: int = 384
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    AttentionClass: Type[nn.Module] = Attention
    FfnClass: Type[nn.Module] = Mlp

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        def attn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = nn.LayerNorm(name="norm1")(x)
            x = self.AttentionClass(
                num_heads=self.num_heads, embed_dim=self.embed_dim, name="attn"
            )(x, training=training)
            x = LayerScale(name="ls1")(x)
            return x

        def ffn_residual_func(x: jnp.ndarray) -> jnp.ndarray:
            x = nn.LayerNorm(name="norm2")(x)
            x = self.FfnClass(
                hidden_features=int(self.mlp_ratio * self.embed_dim),
                out_features=self.embed_dim,
                name="mlp",
            )(x, training=training)
            x = LayerScale(name="ls2")(x)
            return x

        if training:
            x = x + DropPath(
                rate=self.drop_path_rate, name="drop_path1", deterministic=not training
            )(attn_residual_func(x))
            x = x + DropPath(
                rate=self.drop_path_rate, name="drop_path2", deterministic=not training
            )(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)

        return x


class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 14
    in_channels: int = 3
    embed_dim: int = 384
    norm_layer: Type[nn.Module] = None
    flatten_embedding: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, H, W, C = x.shape
        patch_H, patch_W = self.patch_size, self.patch_size
        assert (
            H % patch_H == 0 and W % patch_W == 0
        ), f"Image size ({H}*{W}) cannot be evenly divided by patch size ({patch_H}*{patch_W})."

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(patch_H, patch_W),
            strides=(patch_H, patch_W),
            name="proj",
            padding="VALID",
        )(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm_layer is not None:
            x = self.norm_layer(name="norm")(x)

        if not self.flatten_embedding:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x


class DinoViT(nn.Module):
    img_size: int = 224
    in_channels: int = 3

    patch_size: int = 14
    embed_dim: int = 384

    depth: int = 12

    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    BlockClass: Type[nn.Module] = Block
    AttentionClass: Type[nn.Module] = Attention
    FfnClass: Type[nn.Module] = Mlp
    EmbedLayer: Type[nn.Module] = PatchEmbed

    def _interpolate_pos_encoding(
        self, x: jnp.ndarray, w: int, h: int, pos_embed: jnp.ndarray
    ):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed

        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim),
            (1, w0, h0, dim),
            method="bicubic",
        )
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, -1, dim))

        return jnp.concatenate((class_pos_embed[None], patch_pos_embed), axis=1).astype(
            previous_dtype
        )

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, H, W, C = x.shape
        assert H == W == self.img_size, "x size must be (B, {}, {}, {})".format(
            self.img_size, self.img_size, C
        )

        x = self.EmbedLayer(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            name="patch_embed",
        )(x)
        cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.embed_dim)
        )
        cls_token = jnp.broadcast_to(cls_token, (x.shape[0], *cls_token.shape[1:]))
        x = jnp.concatenate((cls_token, x), axis=1)

        num_patches = (self.img_size // self.patch_size) ** 2
        num_tokens = 1

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + num_tokens, self.embed_dim),
        )
        x = x + self._interpolate_pos_encoding(
            x, self.img_size, self.img_size, pos_embed
        )

        for i in range(self.depth):
            x = self.BlockClass(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop_path_rate=self.drop_path_rate,
                AttentionClass=self.AttentionClass,
                FfnClass=self.FfnClass,
                name=f"blocks.{i}",
            )(x, training=training)

        x_norm = nn.LayerNorm(name="norm")(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }

