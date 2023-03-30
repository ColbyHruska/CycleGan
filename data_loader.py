from PIL import Image
import os
import numpy as np

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.a_dir = f"./datasets/{dataset_name}/trainA/"
        self.a_size = len(os.listdir(self.a_dir))
        self.b_dir = f"./datasets/{dataset_name}/trainB/"
        self.b_size = len(os.listdir(self.b_dir))

    def load_data(self, domain, batch_size=1):
        size = self.a_size if domain == "A" else self.b_size
        dir = self.a_dir if domain == "A" else self.b_dir
        batch_images = np.random.randint(0, size, size=batch_size)

        imgs = []
        for idx in batch_images:
            img_path = f"{dir}{idx}.png"
            img = self.imread(img_path)
            if np.random.random() > 0.5:
                img = np.fliplr(img)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batches(self, batch_size=1):
        self.n_batches = min(self.a_size, self.b_size) // batch_size
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        a_idx = np.random.randint(0, self.a_size, (self.n_batches, batch_size))
        b_idx = np.random.randint(0, self.b_size, (self.n_batches, batch_size))

        for i in range(self.n_batches):
            batch_A = a_idx[i]
            batch_B = b_idx[i]
            imgs_A, imgs_B = [], []
            for a, b in zip(batch_A, batch_B):
                img_A = self.imread(f"{self.a_dir}{a}.png")
                img_B = self.imread(f"{self.b_dir}{b}.png")

                if np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        with open(path, 'rb') as f:
            return np.array(Image.open(f))
