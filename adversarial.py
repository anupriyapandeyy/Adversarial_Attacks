# FGSM Adversarial Attack
atk = FGSM(model, eps=0.03)
labels = predicted  # Using predicted label as dummy
adv_images = atk(image_tensor, labels)

# Predict 
adv_outputs = model(adv_images)
_, adv_predicted = torch.max(adv_outputs.data, 1)
fgsm_prediction = imagenet_labels[adv_predicted.item()]

print(f"Adversarial Prediction: {imagenet_labels[adv_predicted.item()]}")
imshow(adv_images[0], f"Adversarial: {imagenet_labels[adv_predicted.item()]}")
