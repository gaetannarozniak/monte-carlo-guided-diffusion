import matplotlib.pyplot as plt

def plot_image(tensor_image):
    if tensor_image.shape[0] == 1:
        tensor_image = tensor_image.squeeze(0)
    # renormalize between 0 and 1
    tensor_image = (tensor_image*0.5)+0.5
    tensor_image = tensor_image.permute(1, 2, 0)
    np_image = tensor_image.detach().cpu().numpy()
    plt.imshow(np_image)
    plt.show()