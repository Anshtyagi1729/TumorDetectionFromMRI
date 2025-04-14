from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    return transform(image)

# Mapping from index to class (or load from file)
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
