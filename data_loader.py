from PIL import Image
import os
import numpy as np

class DataLoader():
    def __init__(self, src, dst, img_res=(128, 128)):
        self.dataset_name = f"{src}2{dst}"
        self.img_res = img_res
        self.a_dir = f"./datasets/{src}/"
        self.a_size = len(os.listdir(self.a_dir))
        self.b_dir = f"./datasets/{dst}/"
        self.b_size = len(os.listdir(self.b_dir))

    def load_data(self, domain, batch_size=1):
        size = self.a_size if domain == "A" else self.b_size
        dir = self.a_dir if domain == "A" else self.b_dir
        batch_images = np.random.randint(0, size, size=batch_size)

        imgs = []
        while len(imgs) < batch_size:
            try:
                idx = np.random.randint(0, size)
                img_path = f"{dir}{idx}.png"
                img = self.imread(img_path)
                if np.random.random() > 0.5:
                    img = np.fliplr(img)
                imgs.append(img)
            except KeyboardInterrupt:
                raise
            except:
                pass

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batches(self, batch_size=1):
        self.n_batches = min(self.a_size, self.b_size) // batch_size

        for i in range(self.n_batches):
            imgs_A = self.load_data("A", batch_size)
            imgs_B = self.load_data("B", batch_size)

            yield imgs_A, imgs_B

    def imread(self, path):
        with open(path, 'rb') as f:
            return np.array(Image.open(f))
