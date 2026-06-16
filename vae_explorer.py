import argparse, torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

def load_vae():
    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")

def encode(vae, path):
    t = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    img = t(Image.open(path).convert("RGB")).unsqueeze(0).cuda().half()
    with torch.no_grad():
        return vae.encode(img).latent_dist.sample()

def decode(vae, latent):
    with torch.no_grad():
        img = vae.decode(latent).sample
    return transforms.ToPILImage()((img/2+0.5).clamp(0,1).squeeze(0).cpu())

def interp(vae, p1, p2, steps=10):
    l1, l2 = encode(vae, p1), encode(vae, p2)
    return [decode(vae, l1*(1-i/steps) + l2*(i/steps)) for i in range(steps+1)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encode")
    p.add_argument("--interpolate", nargs=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--output", default="output")
    a = p.parse_args()
    vae = load_vae()
    if a.encode:
        torch.save(encode(vae, a.encode), f"{a.output}.pt")
        print(f"Saved: {a.output}.pt")
    elif a.interpolate:
        imgs = interp(vae, a.interpolate[0], a.interpolate[1], a.steps)
        for i, img in enumerate(imgs):
            img.save(f"{a.output}_{i:03d}.png")
        print(f"Saved {len(imgs)} frames")

if __name__ == "__main__":
    main()
