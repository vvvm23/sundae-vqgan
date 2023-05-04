import torch


@torch.inference_mode()
@torch.cuda.amp.autocast()
def decode_vqgan(q, vqgan, latent_shape):
    embeddings = vqgan.quantize.embedding(q)
    embeddings = embeddings.view(q.shape[0], *latent_shape).permute(0, 3, 1, 2)
    img = vqgan.decode(embeddings)
    img = torch.clamp(img, -1.0, 1.0)
    img = (img + 1.0) / 2.0

    return img


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x
