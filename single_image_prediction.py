# Single Image Prediction - Clean Interface
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os

def predict_and_visualize_image(image_path, model, transform, device, class_labels):
    """
    Predict and visualize results for a single image
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return None
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Successfully loaded image: {image_path}")
        print(f"📏 Original image size: {image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Prepare image for model
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # Prepare results
    predicted_class = class_labels[predicted_class_idx]
    all_probs = {class_labels[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display original image
    ax1.imshow(image)
    ax1.set_title(f'Input Image\nSize: {image.size}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display prediction results as bar chart
    classes = list(all_probs.keys())
    probabilities_list = list(all_probs.values())
    colors = ['red' if i == predicted_class_idx else 'lightblue' for i in range(len(classes))]
    
    bars = ax2.bar(classes, probabilities_list, color=colors)
    ax2.set_title(f'Prediction Results\nPredicted: {predicted_class}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities_list):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*60)
    print("🧠 BRAIN TUMOR CLASSIFICATION RESULTS")
    print("="*60)
    print(f"📁 Image Path: {image_path}")
    print(f"🎯 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"🔍 Model: Swin Transformer")
    
    print(f"\n📈 All Class Probabilities:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, prob) in enumerate(sorted_probs):
        status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"   {status} {class_name.replace('_', ' ').title()}: {prob:.2%}")
    
    print("="*60)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }

print("✅ Prediction function defined successfully!")

# Load the best trained model and make prediction on your image
print("🔄 Loading the best trained model...")
try:
    # Load the saved model checkpoint
    checkpoint = torch.load('best_swin_transformer_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded successfully!")
    print(f"📊 Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
except FileNotFoundError:
    print("❌ Error: Model file 'best_swin_transformer_model.pth' not found.")
    print("Please make sure you have trained the model first by running all previous cells.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Test with your specific image
print("\n🖼️  Testing with your image...")
image_path = "image.png"  # Your specified image path

# Run prediction
result = predict_and_visualize_image(
    image_path=image_path,
    model=model,
    transform=val_transforms,
    device=device,
    class_labels=['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
)

if result:
    print(f"\n🎉 Prediction completed successfully!")
    print(f"💡 The model predicts: {result['predicted_class']} with {result['confidence']:.1%} confidence")
else:
    print("\n❌ Prediction failed. Please check the image path and try again.")

# Additional functionality: Test with any custom image path
print("\n" + "="*60)
print("🔧 CUSTOM IMAGE TESTING")
print("="*60)
print("You can test any image by changing the 'custom_image_path' variable below:")
print()

# Uncomment and modify the path below to test other images
# custom_image_path = "path/to/your/custom/image.jpg"
# custom_result = predict_and_visualize_image(
#     image_path=custom_image_path,
#     model=model,
#     transform=val_transforms,
#     device=device,
#     class_labels=['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
# )

print("📝 Instructions:")
print("1. Uncomment the lines above")
print("2. Replace 'path/to/your/custom/image.jpg' with your image path")
print("3. Run the cell to get predictions for your custom image")
print("\n✨ The model will show both the image and prediction results with confidence scores!")