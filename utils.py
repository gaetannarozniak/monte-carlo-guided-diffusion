import matplotlib.pyplot as plt
import os

def plot_image(tensor_image, save_dir="results", filename="image.png", show=False):
    os.makedirs(save_dir, exist_ok=True)  # make directory if it doesn't exist

    if tensor_image.shape[0] == 1:
        tensor_image = tensor_image.squeeze(0)

    # renormalize from [-1, 1] to [0, 1]
    tensor_image = (tensor_image * 0.5) + 0.5
    tensor_image = tensor_image.permute(1, 2, 0)
    np_image = tensor_image.detach().cpu().numpy()

    plt.imshow(np_image)
    plt.axis("off")
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    else:
        plt.close()
